import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.ITCBLModel import ITCBL
from preprocessing.prepare_data import tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load Model
# -----------------------------
model = ITCBL(vocab_size=10000).to(device)
model.load_state_dict(torch.load("models/sqli_model.pth", map_location=device))
model.eval()


# -----------------------------
# Adversarial Generator (ATTACKS ONLY)
# -----------------------------
def generate_adversarial(query):
    return [
        query,
        query.replace(" ", "/**/"),
        query.replace(" ", "%20"),
        query + "--",
        query.replace("1=1", "'1'='1'"),
    ]


# -----------------------------
# Suspicious Pattern Check
# -----------------------------
def has_sqli_pattern(query):
    patterns = ["or", "union", "--", "select", "drop", "'"]
    query = query.lower()
    return any(p in query for p in patterns)


# -----------------------------
# Test Queries (Balanced)
# -----------------------------
test_queries = [
    # NORMAL
    ("GET /home", 0),
    ("GET /products?id=10", 0),
    ("POST /login username=dev password=123", 0),
    ("GET /search?q=shoes", 0),
    ("GET /profile?id=25", 0),

    # ATTACKS
    ("GET /login?id=1 OR 1=1", 1),
    ("GET /search?q=admin'--", 1),
    ("GET /product?id=10 UNION SELECT username,password FROM users", 1),
]


# -----------------------------
# SETTINGS
# -----------------------------
THRESHOLD = 0.8


# -----------------------------
# Metrics
# -----------------------------
total = 0
correct = 0
detected_attacks = 0
total_attacks = 0

false_positives = 0
false_negatives = 0

print("\n========== FINAL ROBUST EVALUATION ==========\n")

for base_query, label in test_queries:

    print(f"\n🔹 Base Query: {base_query}")

    # Apply adversarial only to attacks
    if label == 1:
        query_set = generate_adversarial(base_query)
    else:
        query_set = [base_query]

    for query in query_set:

        encoded = tokenizer.encode(query).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(encoded)
            probs = torch.softmax(output, dim=1)[0]

        attack_prob = probs[1].item()

        # -----------------------------
        # FINAL DECISION LOGIC
        # -----------------------------
        if attack_prob > THRESHOLD and has_sqli_pattern(query):
            pred = 1
        else:
            pred = 0

        total += 1

        # Accuracy
        if pred == label:
            correct += 1

        # Attack detection
        if label == 1:
            total_attacks += 1
            if pred == 1:
                detected_attacks += 1
            else:
                false_negatives += 1

        # False positives
        if label == 0 and pred == 1:
            false_positives += 1
            print("⚠ False Positive:", query)

        status = "✔ Correct" if pred == label else "❌ Wrong"

        print(f"\nQuery: {query}")
        print(f"Prediction: {'SQL Injection' if pred==1 else 'Normal'}")
        print(f"Confidence: {attack_prob:.4f}")
        print(f"Result: {status}")


# -----------------------------
# Final Metrics
# -----------------------------
accuracy = correct / total * 100

if total_attacks > 0:
    detection_rate = detected_attacks / total_attacks * 100
else:
    detection_rate = 0

print("\n========== FINAL RESULTS ==========")
print(f"Total samples: {total}")
print(f"Accuracy: {accuracy:.2f}%")
print(f"Attack Detection Rate: {detection_rate:.2f}%")
print(f"False Positives: {false_positives}")
print(f"False Negatives: {false_negatives}")