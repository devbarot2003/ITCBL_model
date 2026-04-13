import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.ITCBLModel import ITCBL
from preprocessing.prepare_data import tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# LOAD TWO MODELS
# -----------------------------
model_no_attn = ITCBL(vocab_size=10000).to(device)
model_no_attn.load_state_dict(torch.load("models/sqli_model_no_attention.pth", map_location=device))
model_no_attn.eval()

model_attn = ITCBL(vocab_size=10000).to(device)
model_attn.load_state_dict(torch.load("models/sqli_model_attention.pth", map_location=device))
model_attn.eval()


# -----------------------------
# TEST DATA
# -----------------------------
test_queries = [
    ("GET /home", 0),
    ("GET /products?id=10", 0),
    ("GET /login?id=1 OR 1=1", 1),
    ("GET /search?q=admin'--", 1),
    ("GET /product?id=10 UNION SELECT username,password FROM users", 1),
]


THRESHOLD = 0.8


def evaluate(model):

    total = 0
    correct = 0
    false_pos = 0
    detected = 0
    total_attacks = 0

    for query, label in test_queries:

        encoded = tokenizer.encode(query).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(encoded)
            probs = torch.softmax(output, dim=1)[0]

        attack_prob = probs[1].item()

        pred = 1 if attack_prob > THRESHOLD else 0

        total += 1

        if pred == label:
            correct += 1

        if label == 1:
            total_attacks += 1
            if pred == 1:
                detected += 1

        if label == 0 and pred == 1:
            false_pos += 1

    accuracy = correct / total * 100
    detection = detected / total_attacks * 100

    return accuracy, detection, false_pos


# -----------------------------
# RUN COMPARISON
# -----------------------------
acc1, det1, fp1 = evaluate(model_no_attn)
acc2, det2, fp2 = evaluate(model_attn)

print("\n========== MODEL COMPARISON ==========")
print(f"Without Attention → Accuracy: {acc1:.2f}%, Detection: {det1:.2f}%, FP: {fp1}")
print(f"With Attention    → Accuracy: {acc2:.2f}%, Detection: {det2:.2f}%, FP: {fp2}")