import torch
import torch.nn as nn

from ImgProcessing import ImageEncoder, CaptionDecoder

class ImageToText(nn.Module):
    """
    Full model pipelining CNN Encoder and LSTM Decoder.
    """
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, max_caption_length=20):
        super(ImageToText, self).__init__()
        self.encoder = ImageEncoder(embed_size)
        self.decoder = CaptionDecoder(embed_size, hidden_size, vocab_size, num_layers)
        self.max_caption_length = max_caption_length

    def forward(self, images):
        """
        If captions are provided, it trains the model.
        Otherwise, it generates a caption from an image.
        """
        features = self.encoder(images)  
        return self.generate_caption(features)

    def generate_caption(self, features, start_token=1, end_token=2):
        batch_size = features.shape[0]
        generated_captions = []

        for i in range(batch_size):
            generated_caption = []            
            hidden = None  # LSTM hidden state
            
            token = torch.tensor([[start_token]], device=features.device)  # (1,1)

            for _ in range(self.max_caption_length):
                embedding = self.decoder.embed(token)  # (1,1,embed_size)
                
                lstm_out, hidden = self.decoder.lstm(embedding, hidden)  # Pass embedding directly
                outputs = self.decoder.fc(lstm_out.squeeze(1))  
                
                token = outputs.argmax(dim=1, keepdim=True)  
                generated_caption.append(token.item())  
                
                if token.item() == end_token:
                    break
            
            generated_captions.append(generated_caption)  
        
        return generated_captions  
