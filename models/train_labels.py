import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import json
import random
from collections import Counter
import os
from GenerateLabel import BiRNNClusterClassifier
from dotenv import load_dotenv
import kagglehub


"""
Trains the generating labels model (GenerateLabel.py)
Info regarding the dataset is included in the paper.
"""
load_dotenv()

BERT_MODEL_NAME = "bert-base-cased"

downloaded_dataset_dir = kagglehub.dataset_download("rmisra/news-category-dataset")

actual_dataset_filename = "News_Category_Dataset_v3.json" 

DATASET_FILE_PATH = os.path.join(downloaded_dataset_dir, actual_dataset_filename)


MODEL_SAVE_PATH = os.getenv('GENERATE_LABEL_PATH', 'GenerateLabel.pt')
VOCAB_SAVE_PATH = "label_vocab.json"

MAX_CLUSTERS_TO_PROCESS = 5000
MIN_CLUSTER_SIZE = 5
MAX_HEADLINES_PER_CLUSTER = 10
NUM_EPOCHS = 10 
LEARNING_RATE = 1e-4
HIDDEN_DIM_RNN = 128

def load_news_data(dataset_file_path, max_clusters=None):
    data_by_category = {}
    processed_categories_count = 0
    
    print(f"Attempting to load dataset from: {dataset_file_path}")
    if not os.path.exists(dataset_file_path):
        print(f"Error: Dataset JSON file not found at {dataset_file_path}")
        alt_filename = "News_Category_Dataset_v2.json" if "v3" in dataset_file_path else "News_Category_Dataset_v3.json"
        alt_dataset_file_path = os.path.join(os.path.dirname(dataset_file_path), alt_filename)
        if os.path.exists(alt_dataset_file_path):
            print(f"Found alternative: {alt_dataset_file_path}. Please update actual_dataset_filename in the script.")
            dataset_file_path = alt_dataset_file_path 
            print(f"Now attempting to load from: {dataset_file_path}")
        else:
            print(f"Alternative {alt_filename} also not found in {os.path.dirname(dataset_file_path)}.")
            return None
            
    try:
        with open(dataset_file_path, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f):
                try:
                    record = json.loads(line)
                    category = record.get('category', '').strip().upper() 
                    headline = record.get('headline', '').strip()

                    if not category or not headline:
                        continue
                    
                    if ' ' in category: 
                        continue

                    if category not in data_by_category:
                        if max_clusters is not None and processed_categories_count >= max_clusters:
                            if category not in data_by_category.keys(): 
                                continue
                        data_by_category[category] = []
                        if category not in data_by_category or len(data_by_category[category]) == 0:
                             processed_categories_count +=1
                    
                    data_by_category[category].append(headline)
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line at line {line_number+1}: {line[:100]}")
                    continue
    except FileNotFoundError:
        print(f"Error: Dataset JSON file still not found at {dataset_file_path}.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while reading the file: {e}")
        return None


    filtered_data = {
        cat: headlines for cat, headlines in data_by_category.items()
        if len(headlines) >= MIN_CLUSTER_SIZE
    }
    
    if max_clusters is not None and len(filtered_data) > max_clusters:
        selected_cats = random.sample(list(filtered_data.keys()), max_clusters)
        filtered_data = {cat: filtered_data[cat] for cat in selected_cats}

    print(f"Loaded {len(filtered_data)} categories with at least {MIN_CLUSTER_SIZE} headlines.")
    return filtered_data

def build_label_vocab(data_by_category):
    all_labels = list(data_by_category.keys())
    if not all_labels:
        raise ValueError("No labels found to build vocabulary. Check data loading and filtering.")
    
    label_counts = Counter(all_labels)
    vocab = sorted(label_counts, key=label_counts.get, reverse=True)
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {str(idx): word for idx, word in enumerate(vocab)} # Ensure idx is string for JSON
    print(f"Built label vocabulary with {len(vocab)} unique labels.")
    return word2idx, idx2word

def get_bert_embeddings_for_training_cluster(headlines, bert_model, tokenizer, device, max_len=128):
    embeddings = []
    if len(headlines) > MAX_HEADLINES_PER_CLUSTER:
        headlines_sample = random.sample(headlines, MAX_HEADLINES_PER_CLUSTER)
    else:
        headlines_sample = headlines
    
    for headline in headlines_sample:
        inputs = tokenizer(headline, return_tensors="pt", truncation=True, padding="max_length", max_length=max_len).to(device)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze()
        embeddings.append(cls_embedding)
    
    if not embeddings:
        return None
    return embeddings

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading BERT model ({BERT_MODEL_NAME}) for embeddings...")
    bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    bert_embed_model = AutoModel.from_pretrained(BERT_MODEL_NAME).to(device)
    bert_embed_model.eval()
    BERT_EMBEDDING_DIM = bert_embed_model.config.hidden_size

    data_by_category = load_news_data(DATASET_FILE_PATH, max_clusters=MAX_CLUSTERS_TO_PROCESS)
    if data_by_category is None or not data_by_category:
        print("Exiting due to data loading failure.")
        exit()

    word2idx, idx2word = build_label_vocab(data_by_category)
    
    vocab_to_save = {
        'word2idx': word2idx,
        'idx2word': idx2word, 
        'embedding_dim': BERT_EMBEDDING_DIM,
        'hidden_dim': HIDDEN_DIM_RNN
    }
    with open(VOCAB_SAVE_PATH, 'w', encoding='utf-8') as f:
        json.dump(vocab_to_save, f)
    print(f"Label vocabulary saved to {VOCAB_SAVE_PATH}")

    print("Preparing training data (this may take a while)...")
    train_data_raw = []
    for i, (category, headlines) in enumerate(data_by_category.items()):
        if i % 100 == 0 and i > 0:
            print(f"  Processed {i}/{len(data_by_category)} categories for training data preparation.")
        if category in word2idx:
            label_idx = word2idx[category]
            cluster_embeddings_list = get_bert_embeddings_for_training_cluster(headlines, bert_embed_model, bert_tokenizer, device)
            if cluster_embeddings_list and len(cluster_embeddings_list) > 0:
                train_data_raw.append((cluster_embeddings_list, label_idx))
    
    if not train_data_raw:
        exit()
    
    print(f"Prepared {len(train_data_raw)} training samples.")

    vocab_size = len(word2idx)
    model = BiRNNClusterClassifier(embedding_dim=BERT_EMBEDDING_DIM, hidden_dim=HIDDEN_DIM_RNN, vocab_size=vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.NLLLoss()

    print("Starting training")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        random.shuffle(train_data_raw)

        for i, (cluster_embeddings, label_idx_val) in enumerate(train_data_raw):
            if not cluster_embeddings:
                continue
            
            input_tensor = torch.stack(cluster_embeddings).unsqueeze(0).to(device)
            label_tensor = torch.tensor([label_idx_val], dtype=torch.long).to(device)
            
            optimizer.zero_grad()
            log_probs = model(input_tensor)
            loss = loss_fn(log_probs, label_tensor)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (i + 1) % 200 == 0: 
                 print(f"  Epoch {epoch+1}/{NUM_EPOCHS}, Batch {i+1}/{len(train_data_raw)}, Loss: {loss.item():.4f}")


        avg_loss = total_loss / len(train_data_raw) if len(train_data_raw) > 0 else 0
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} Loss: {avg_loss:.4f}")
        
        if MODEL_SAVE_PATH:
            torch.save(model.state_dict(), MODEL_SAVE_PATH)


    print("Training finished.")
    if MODEL_SAVE_PATH:
        print(f"Final model saved to {MODEL_SAVE_PATH}")
