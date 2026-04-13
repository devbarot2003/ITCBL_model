import torch
import torch.nn as nn
import torch.nn.functional as F


class ITCBL(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_classes=2):
        super(ITCBL, self).__init__()

        # Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # TextCNN
        self.conv = nn.Conv1d(in_channels=embed_dim, out_channels=128, kernel_size=3, padding=1)

        # BiLSTM
        self.lstm = nn.LSTM(128, hidden_dim, batch_first=True, bidirectional=True)

        # Attention Layer
        self.attention = nn.Linear(hidden_dim * 2, 1)

        # Fully connected
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

        self.dropout = nn.Dropout(0.5)


    def forward(self, x):

        # Embedding
        x = self.embedding(x)  # (batch, seq, embed)

        # CNN expects (batch, channels, seq)
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv(x))

        # Back to (batch, seq, features)
        x = x.permute(0, 2, 1)

        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden*2)

        # -----------------------------
        # ATTENTION MECHANISM
        # -----------------------------
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)  # (batch, seq, 1)

        context_vector = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden*2)

        # -----------------------------
        # Classification
        # -----------------------------
        x = self.dropout(context_vector)
        x = self.fc(x)

        return x