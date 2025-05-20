import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import json
import os

#Model for generating a singular word(output distributed once over vocabulary) to describe a cluster of words
#Uses standard bert embeddings for lightweight capabilities
class BiRNNClusterClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            bidirectional=True,
            batch_first=True,
            num_layers=1 
        )
        self.classifier = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x):
        outputs, (hidden, cell) = self.rnn(x)
        pooled = outputs.mean(dim=1)
        logits = self.classifier(pooled)
        return F.log_softmax(logits, dim=-1)

BERT_MODEL_NAME = "bert-base-cased"
DEFAULT_EMBEDDING_DIM = 768 
DEFAULT_HIDDEN_DIM = 128    

def get_bert_embedding_for_inference(text_list, bert_model, tokenizer, device):
    embeddings = []
    for text in text_list:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze() 
        embeddings.append(cls_embedding)
    if not embeddings:
        return None
    return torch.stack(embeddings).unsqueeze(0) 


def SeqToOne(cluster: list[str], model_path='C:\\Users\\15169\\Documents\\KnowledgeGraph\\models\\GenerateLabel.pt', vocab_path='label_vocab.json'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)
        word2idx = vocab_data['word2idx']
        idx2word = vocab_data['idx2word']
        idx2word = {int(k): v for k, v in idx2word.items()}
        loaded_embedding_dim = vocab_data.get('embedding_dim', DEFAULT_EMBEDDING_DIM)
        loaded_hidden_dim = vocab_data.get('hidden_dim', DEFAULT_HIDDEN_DIM)

    except FileNotFoundError:
        print(f"Error: Vocabulary file not found at {vocab_path}")
        return "Error: Vocab not found"
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {vocab_path}")
        return "Error: Vocab corrupted"


    vocab_size = len(word2idx)
    try:
        bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
        bert_embedder = AutoModel.from_pretrained(BERT_MODEL_NAME).to(device)
        bert_embedder.eval()
    except Exception as e:
        print(f"Error loading BERT model: {e}")
        return "Error: BERT load failed"

    birnn_model = BiRNNClusterClassifier(embedding_dim=loaded_embedding_dim, hidden_dim=loaded_hidden_dim, vocab_size=vocab_size)
    try:
        birnn_model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Trained model not found at {model_path}")
        return "Error: Model not found"
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        return "Error: Model load failed"
        
    birnn_model = birnn_model.to(device)
    birnn_model.eval()

    if not cluster:
        return "Error: Empty cluster"

    with torch.no_grad():
        input_embeddings = get_bert_embedding_for_inference(cluster, bert_embedder, bert_tokenizer, device)
        
        if input_embeddings is None or input_embeddings.nelement() == 0:
            print("Warning: Could not generate valid embeddings for the input cluster.")
            return "Error: Embedding failed"

        log_probs = birnn_model(input_embeddings)
        pred_id = log_probs.argmax(dim=-1).item()
        
        label = idx2word.get(pred_id, "Unknown_Label")

    return label


