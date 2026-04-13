from collections import Counter
import torch

class Tokenizer:
    def __init__(self, max_vocab=10000):
        self.max_vocab = max_vocab
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}

    def build_vocab(self, texts):
        counter = Counter()

        for text in texts:
            counter.update(text.split())

        most_common = counter.most_common(self.max_vocab - 2)

        for idx, (word, _) in enumerate(most_common, start=2):
            self.word2idx[word] = idx

    def encode(self, text, max_len=100):

        tokens = text.split()
        ids = [self.word2idx.get(token, 1) for token in tokens]

        ids = ids[:max_len]
        ids += [0] * (max_len - len(ids))

        return torch.tensor(ids)