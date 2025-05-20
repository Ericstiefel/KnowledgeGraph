import torch
from transformers import AutoModel, AutoTokenizer
from dotenv import load_dotenv
import torch.nn as nn
import pandas as pd
import random
from LLMJudge import LogisticRegressionTorch

"""
File for training the LLMJudge
Determines if proposed cluster is confirmed or denied
Info about the dataset used can be found in the connected paper.
"""

load_dotenv()

def load_bert_model_and_tokenizer(model_name="bert-base-cased"):
    print(f"Loading BERT tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Loading BERT model: {model_name}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    print(f"BERT model and tokenizer loaded successfully on {device}.")
    return model, tokenizer, device

def get_bert_embedding(text, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding.cpu().squeeze()

def cluster_features_bert(entities, model, tokenizer, device):
    if not entities or len(entities) < 1:
        return torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])

    embeddings = []
    for e_text in entities:
        if not isinstance(e_text, str) or not e_text.strip():
            continue
        try:
            emb = get_bert_embedding(e_text, model, tokenizer, device)
            if emb.ndim > 0 and emb.numel() > 0 :
                 embeddings.append(emb)
        except Exception:
            continue
    
    if len(embeddings) < 2:
        return torch.tensor([0.0, 0.0, 0.0, 0.0, float(len(entities))])
        
    stacked_embeddings = torch.stack(embeddings)
    
    sim_matrix = torch.nn.functional.cosine_similarity(
        stacked_embeddings.unsqueeze(1), stacked_embeddings.unsqueeze(0), dim=-1
    )

    if sim_matrix.shape[0] < 2:
         return torch.tensor([0.0, 0.0, 0.0, 0.0, float(len(entities))])

    upper_tri_indices = torch.triu_indices(sim_matrix.shape[0], sim_matrix.shape[1], offset=1)
    
    if upper_tri_indices.numel() == 0:
        mean_sim, std_sim, max_sim, min_sim = 0.0, 0.0, 0.0, 0.0
        if sim_matrix.numel() > 0 :
             val = sim_matrix.item() if sim_matrix.numel() == 1 else 0.0
             mean_sim, max_sim, min_sim = val,val,val
    else:
        upper_tri = sim_matrix[upper_tri_indices[0], upper_tri_indices[1]]
        if upper_tri.numel() == 0:
            mean_sim, std_sim, max_sim, min_sim = 0.0, 0.0, 0.0, 0.0
        else:
            mean_sim = upper_tri.mean().item()
            std_sim = upper_tri.std().item() if upper_tri.numel() > 1 else 0.0
            max_sim = upper_tri.max().item()
            min_sim = upper_tri.min().item()

    if torch.isnan(torch.tensor(std_sim)): 
        std_sim = 0.0
    
    return torch.tensor([
        mean_sim,
        std_sim,
        max_sim,
        min_sim,
        float(len(entities))
    ])

def load_and_prepare_uci_product_data(csv_path, title_col='PRODUCT_NAME', cluster_col='CLUSTER_ID', min_cluster_size=2):
    print(f"Loading UCI product data")
    try:
        df = pd.read_csv(csv_path, on_bad_lines='skip')
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

    if title_col not in df.columns or cluster_col not in df.columns:
        return None
    
    df = df.dropna(subset=[title_col, cluster_col])
    df = df[df[title_col].apply(lambda x: isinstance(x, str) and len(x.strip()) > 0)]
    
    cluster_counts = df[cluster_col].value_counts()
    valid_clusters = cluster_counts[cluster_counts >= min_cluster_size].index
    df_filtered = df[df[cluster_col].isin(valid_clusters)]
    
    print(f"Loaded {len(df_filtered)} products from {df_filtered[cluster_col].nunique()} valid clusters (min size {min_cluster_size}).")
    return df_filtered

def create_labeled_clusters_from_df(df, title_col, cluster_col, num_samples=1000, items_per_cluster_range=(2, 5)):
    if df is None or df.empty:
        return []
        
    labeled_clusters = []
    grouped_by_cluster = df.groupby(cluster_col)[title_col].apply(list).to_dict()
    cluster_ids = list(grouped_by_cluster.keys())

    if not cluster_ids or len(cluster_ids) < 2 :
        if not cluster_ids: return []

    for i in range(num_samples):
        num_items = random.randint(items_per_cluster_range[0], items_per_cluster_range[1])
        
        if i % 2 == 0 and cluster_ids: 
            chosen_cluster_id = random.choice(cluster_ids)
            products_in_cluster = grouped_by_cluster[chosen_cluster_id]
            if len(products_in_cluster) >= num_items:
                cluster_entities = random.sample(products_in_cluster, num_items)
                labeled_clusters.append((cluster_entities, 1))
            elif len(products_in_cluster) >= items_per_cluster_range[0]:
                cluster_entities = random.sample(products_in_cluster, len(products_in_cluster))
                labeled_clusters.append((cluster_entities, 1))
        else: 
            cluster_entities = []
            if len(cluster_ids) >= num_items:
                selected_cluster_ids_for_neg = random.sample(cluster_ids, num_items)
                for k_idx in range(num_items):
                    prod_list = grouped_by_cluster[selected_cluster_ids_for_neg[k_idx]]
                    if prod_list:
                         cluster_entities.append(random.choice(prod_list))
            elif cluster_ids:
                for _ in range(num_items):
                    random_product_title = df[title_col].sample(1).iloc[0]
                    cluster_entities.append(random_product_title)
            
            if len(cluster_entities) >= items_per_cluster_range[0]:
                 labeled_clusters.append((cluster_entities, 0))
                 
    print(f"Generated {len(labeled_clusters)} labeled cluster examples.")
    return labeled_clusters

if __name__ == '__main__':
    bert_model, bert_tokenizer, device = load_bert_model_and_tokenizer()

    dataset_csv_path = 'Products.csv' 
    product_title_column = 'PRODUCT_NAME'
    cluster_id_column = 'CLUSTER_ID'
    
    product_df = load_and_prepare_uci_product_data(
        dataset_csv_path, 
        product_title_column, 
        cluster_id_column
    )
    
    if product_df is not None and not product_df.empty:
        labeled_clusters = create_labeled_clusters_from_df(
            product_df, 
            product_title_column, 
            cluster_id_column,
            num_samples=2000, 
            items_per_cluster_range=(2, 5)
        )
    
    if not labeled_clusters:
        exit()

    X = []
    y = []
    print("Processing clusters for training...")
    for i, (cluster, label) in enumerate(labeled_clusters):
        if i % 100 == 0:
            print(f"Processing sample {i}/{len(labeled_clusters)}")
        features = cluster_features_bert(cluster, bert_model, bert_tokenizer, device)
        X.append(features)
        y.append(torch.tensor([label], dtype=torch.float))

    X = torch.stack(X).to(device)
    y = torch.stack(y).to(device)

    model_lr = LogisticRegressionTorch(input_dim=X.shape[1]).to(device)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model_lr.parameters(), lr=1e-3)

    print("Starting training")
    num_epochs = 200
    for epoch in range(num_epochs):
        model_lr.train()
        optimizer.zero_grad()
        
        preds = model_lr(X).squeeze(-1)
        current_y = y.squeeze(-1)
        loss = loss_fn(preds, current_y)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0 or epoch == num_epochs -1:
            print(f"Epoch {epoch} - Loss: {loss.item():.4f}")
            
    torch.save(model_lr.state_dict(), 'OutputLabel_BERT.pt')
    print("Training complete")


