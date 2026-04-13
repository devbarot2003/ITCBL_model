# import pandas as pd
# import re
# from sklearn.model_selection import train_test_split
# from torch.utils.data import DataLoader
# from preprocessing.tokenizer import Tokenizer
# from preprocessing.dataset import SQLiDataset
# import torch
# def clean_text(text):

#     text = str(text).lower()
#     text = re.sub(r"http[s]?://", "", text)
#     text = re.sub(r"[^a-zA-Z0-9_=./?-]", " ", text)
#     text = re.sub(r"\s+", " ", text).strip()

#     return text


# df = pd.read_csv("data/csic_database.csv")

# # combine URL + POST content
# df["content"] = df["content"].fillna("")
# df["input_text"] = df["URL"] + " " + df["content"]

# X = df["input_text"].apply(clean_text)
# y = df["classification"]

# X_train, X_temp, y_train, y_temp = train_test_split(
#     X, y, test_size=0.3, random_state=42
# )

# X_val, X_test, y_val, y_test = train_test_split(
#     X_temp, y_temp, test_size=0.5, random_state=42
# )

# print("Train size:", len(X_train))
# print("Validation size:", len(X_val))
# print("Test size:", len(X_test))


# tokenizer = Tokenizer(max_vocab=10000)
# tokenizer.build_vocab(X_train)

# print("Vocabulary size:", len(tokenizer.word2idx))


# train_dataset = SQLiDataset(X_train, y_train, tokenizer)
# val_dataset = SQLiDataset(X_val, y_val, tokenizer)
# test_dataset = SQLiDataset(X_test, y_test, tokenizer)


# train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=128)
# test_loader = DataLoader(test_dataset, batch_size=128)
import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.dataset import SQLiDataset
from preprocessing.tokenizer import Tokenizer


# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("data/csic_database.csv")

# Fill missing content
df["content"] = df["content"].fillna("")

# Combine request fields into one text input
df["input_text"] = (
    df["Method"].astype(str) + " " +
    df["URL"].astype(str) + " " +
    df["content"].astype(str)
)

texts = df["input_text"]
labels = df["classification"]


# -----------------------------
# Initialize Tokenizer
# -----------------------------
tokenizer = Tokenizer()
tokenizer.build_vocab(texts)


# -----------------------------
# Train / Validation / Test Split
# -----------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    texts,
    labels,
    test_size=0.3,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42
)


# -----------------------------
# Create Datasets
# -----------------------------
train_dataset = SQLiDataset(X_train, y_train, tokenizer)
val_dataset = SQLiDataset(X_val, y_val, tokenizer)
test_dataset = SQLiDataset(X_test, y_test, tokenizer)


# -----------------------------
# Create DataLoaders
# -----------------------------
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


print("Data preparation complete.")
print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")