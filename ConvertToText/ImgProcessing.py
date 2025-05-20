import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence



class ImageEncoder(nn.Module):
    def __init__(self, feature_dim):
        super(ImageEncoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(64, feature_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1) 
        x = self.fc(x)
        return x

class CaptionDecoder(nn.Module):
    """
    LSTM-based decoder that generates text from image features.
    """
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(CaptionDecoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features, captions, lengths):
        embeddings = self.embed(captions)  
        inputs = torch.cat((features.unsqueeze(1), embeddings), 1)  
        packed = pack_padded_sequence(inputs, lengths, batch_first=True, enforce_sorted=False)
        hiddens, _ = self.lstm(packed)
        outputs = self.fc(hiddens[0])  
        return outputs