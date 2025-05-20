import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

#Seemed more simple to just create a Logistic Regression torch model, especially since the text already uses torch architecture.
#Outputs converted into binary decision, determining if the cluster is valid
#Uses standard lightweight BERT embeddings

class LogisticRegressionTorch(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

def embed_entity(text: str) -> torch.Tensor:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    embedder = AutoModel.from_pretrained("bert-base-cased").to(device)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = embedder(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]  
    return cls_embedding.squeeze(0)

def Judge(cluster: list[str]) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load('C:\\Users\\15169\\Documents\\KnowledgeGraph\\models\\EntityDetection.pt') 

    model.eval()
    if not cluster:
        return 0
    
    embeddings = torch.stack([embed_entity(e) for e in cluster], dim=0)  # [N, 768]
    cluster_embedding = embeddings.mean(dim=0).unsqueeze(0)  # [1, 768]

    with torch.no_grad():
        prob = model(cluster_embedding).item()
    
    return True if prob > 0.5 else False
