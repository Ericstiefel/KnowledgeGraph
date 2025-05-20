import torch.nn as nn
from torchcrf import CRF

#This model is trained to detect entities out of text
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dims, num_tags):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dims // 2, num_layers=1, bidirectional=True, batch_first=True
        )
        self.hidden2hidden1 = nn.Linear(hidden_dims, hidden_dims)
        self.relu = nn.ReLU()  
        self.hidden2hidden2 = nn.Linear(hidden_dims, hidden_dims)
        self.fc = nn.Linear(hidden_dims, num_tags)
        self.crf = CRF(num_tags, batch_first=True)

    def forward(self, input_ids, labels=None, mask=None):
        emb = self.embedding(input_ids)
        lstm_out, _ = self.lstm(emb)
        hidden1 = self.hidden2hidden1(lstm_out)
        hidden_relu1 = self.relu(hidden1) 
        hidden2 = self.hidden2hidden2(hidden_relu1) 
        hidden_relu2 = self.relu(hidden2)
        emissions = self.fc(hidden_relu2)
        if labels is not None:
            loss = -self.crf(emissions, labels, mask=mask, reduction="mean")
            return loss
        else:
            prediction = self.crf.decode(emissions, mask=mask)
            return prediction
    