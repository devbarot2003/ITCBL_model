import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import confusion_matrix, classification_report

from models.ITCBLModel import ITCBL
from preprocessing.prepare_data import train_loader, val_loader, tokenizer

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# Model
# -----------------------------
vocab_size = len(tokenizer.word2idx)
model = ITCBL(vocab_size=vocab_size).to(device)

print("\nModel initialized successfully\n")

# -----------------------------
# Loss & Optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# Training Settings
# -----------------------------
epochs = 10

print("========== TRAINING STARTED ==========\n")

# -----------------------------
# Training Loop
# -----------------------------
for epoch in range(epochs):

    model.train()
    total_loss = 0

    correct = 0
    total = 0

    for inputs, labels in train_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Training accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(train_loader)
    train_accuracy = 100 * correct / total

    # -----------------------------
    # Validation + Confusion Matrix
    # -----------------------------
    model.eval()
    val_correct = 0
    val_total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_accuracy = 100 * val_correct / val_total

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Classification Report
    report = classification_report(all_labels, all_preds)

    # -----------------------------
    # Output
    # -----------------------------
    print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {train_accuracy:.2f}%")
    print(f"Validation Accuracy: {val_accuracy:.2f}%\n")

print("Confusion Matrix:")
print(cm)

print("\nClassification Report:")
print(report)

print("\n----------------------------\n")


# -----------------------------
# Save Model
# -----------------------------
os.makedirs("models", exist_ok=True)

torch.save(model.state_dict(), "models/sqli_model.pth")

print("========== TRAINING COMPLETED ==========")
print("Model saved at: models/sqli_model.pth")