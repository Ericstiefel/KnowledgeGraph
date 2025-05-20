from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import time
from EntityDetection import BiLSTM_CRF
import torch


"""
Training for Entity Detection model at EntityDetection.py
Info about the dataset conll2003 can be located in the paper.

"""

dataset = load_dataset("conll2003")
label_list = dataset["train"].features["ner_tags"].feature.names
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for l, i in label2id.items()}


tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
vocab_size = tokenizer.vocab_size

class NERDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=32):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]["tokens"]
        labels = self.data[idx]["ner_tags"]
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids = token_ids[:self.max_len] + [0] * (self.max_len - len(token_ids))
        label_ids = labels[:self.max_len] + [0] * (self.max_len - len(labels))
        mask = [1] * min(len(tokens), self.max_len) + [0] * (self.max_len - len(tokens))
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(label_ids, dtype=torch.long),
            torch.tensor(mask, dtype=torch.bool),
        )


train_dataset = NERDataset(dataset["train"], tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTM_CRF(
        vocab_size, embedding_dim=128, hidden_dims=256, num_tags=len(label_list)
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    
    start_time = time.time()
    end_time = start_time + 3.5 * 3600 # 3.5 hours training time

    epoch = 0
    #while time.time() < end_time:
    for _ in range(10):
        total_loss = 0
        for input_ids, labels, mask in tqdm(train_loader):
            input_ids, labels, mask = input_ids.to(device), labels.to(device), mask.to(device)

            loss = model(input_ids, labels, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1} Loss: {total_loss / len(train_loader):.4f}")
        torch.save(model.state_dict(), 'C:\\Users\\15169\\Documents\\KnowledgeGraph\\models\\EntityDetection.pt') 
        epoch += 1

    print("Training Complete!")

    