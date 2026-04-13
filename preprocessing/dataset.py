import torch
from torch.utils.data import Dataset

class SQLiDataset(Dataset):

    def __init__(self, texts, labels, tokenizer, max_len=100):

        self.texts = texts.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        text = self.texts[idx]
        label = self.labels[idx]

        encoded = self.tokenizer.encode(text, self.max_len)

        return encoded, torch.tensor(label, dtype=torch.long)