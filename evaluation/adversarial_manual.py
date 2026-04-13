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
# Suspicious Pattern Check
# -----------------------------
def has_sqli_pattern(query):
    patterns = ["or", "union", "--", "select", "drop", "'"]
    query = query.lower()
    return any(p in query for p in patterns)


# -----------------------------
# Test Queries (Manual + Realistic)
# -----------------------------
test_queries = [

    # NORMAL
    ("GET /home", 0),
    ("GET /products?id=10", 0),
    ("POST /login username=dev password=123", 0),
    ("GET /search?q=shoes", 0),
    ("GET /profile?id=25", 0),

    # BASIC ATTACKS
    ("GET /login?id=1 OR 1=1", 1),
    ("GET /search?q=admin'--", 1),
    ("GET /product?id=10 UNION SELECT username,password FROM users", 1),

    # ADVERSARIAL ATTACKS
    ("GET /login?id=1/**/OR/**/1=1", 1),
    ("GET /login?id=1%20OR%201=1", 1),
    ("GET /login?id=1 OR '1'='1'", 1),
    ("GET /search?q=admin' OR '1'='1'--", 1),
    ("GET /product?id=10 UNION/**/SELECT/**/user,password", 1),
    ("GET /login?id=1 oR 1=1", 1),
    ("GET /login?id=1 OR 1=1#", 1),
]


THRESHOLD = 0.8


# -----------------------------
# FUNCTION: AUTO EVALUATION
# -----------------------------
def run_auto_test():
    total = 0
    correct = 0
    detected_attacks = 0
    total_attacks = 0
    false_positives = 0
    false_negatives = 0

    print("\n========== AUTO EVALUATION ==========\n")

    for query, label in test_queries:

        print(f"\n🔹 Query: {query}")

        encoded = tokenizer.encode(query).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(encoded)
            probs = torch.softmax(output, dim=1)[0]

        attack_prob = probs[1].item()

        if attack_prob > THRESHOLD and has_sqli_pattern(query):
            pred = 1
        else:
            pred = 0

        total += 1

        if pred == label:
            correct += 1

        if label == 1:
            total_attacks += 1
            if pred == 1:
                detected_attacks += 1
            else:
                false_negatives += 1

        if label == 0 and pred == 1:
            false_positives += 1
            print("⚠ False Positive detected!")

        status = "✔ Correct" if pred == label else "❌ Wrong"

        print(f"Prediction: {'SQL Injection' if pred==1 else 'Normal'}")
        print(f"Confidence: {attack_prob:.4f}")
        print(f"Result: {status}")

    accuracy = correct / total * 100
    detection_rate = (detected_attacks / total_attacks * 100) if total_attacks > 0 else 0

    print("\n========== FINAL RESULTS ==========")
    print(f"Total samples: {total}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Attack Detection Rate: {detection_rate:.2f}%")
    print(f"False Positives: {false_positives}")
    print(f"False Negatives: {false_negatives}")


# -----------------------------
# FUNCTION: MANUAL INPUT
# -----------------------------
def run_manual_test():
    print("\n========== MANUAL TEST MODE ==========\n")

    while True:
        query = input("Enter SQL query (or type 'exit'): ")

        if query.lower() == "exit":
            break

        encoded = tokenizer.encode(query).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(encoded)
            probs = torch.softmax(output, dim=1)[0]

        attack_prob = probs[1].item()

        if attack_prob > THRESHOLD and has_sqli_pattern(query):
            pred = 1
        else:
            pred = 0

        print("\n--- RESULT ---")
        print(f"Prediction: {'SQL Injection' if pred==1 else 'Normal'}")
        print(f"Confidence: {attack_prob:.4f}")


# -----------------------------
# MAIN MENU
# -----------------------------
print("\nSelect Mode:")
print("1 → Auto Evaluation (dataset test)")
print("2 → Manual Input (real-time testing)")

choice = input("Enter choice: ")

if choice == "1":
    run_auto_test()
elif choice == "2":
    run_manual_test()
else:
    print("Invalid choice")