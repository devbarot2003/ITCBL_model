import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.ITCBLModel import ITCBL
from preprocessing.prepare_data import tokenizer
from security.session_tracker import SessionTracker

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ITCBL(vocab_size=10000).to(device)
model.load_state_dict(torch.load("models/sqli_model.pth", map_location=device))
model.eval()

tracker = SessionTracker()

# -----------------------------
# TEST SCENARIOS
# -----------------------------
test_cases = {
    "Test Case 1 (Normal Request)": [
        "GET /home?id=5"
    ],

    "Test Case 2 (Direct SQL Injection)": [
        "GET /login?id=1 OR 1=1 --"
    ],

    "Test Case 3 (Second-Order SQL Injection)": [
        "username=admin'--",       # stored malicious
        "GET /profile?id=10"       # later request
    ],

    "Test Case 4 (Obfuscated SQL Injection)": [
        "GET /login?id=1 OR/**/1=1"
    ]
}

# -----------------------------
# RUN TESTS
# -----------------------------
for test_name, requests in test_cases.items():

    print("\n==============================")
    print(test_name)
    print("==============================")

    session_id = test_name  # unique session per test

    for req in requests:

        print(f"\nRequest: {req}")

        tracker.store_request(session_id, req)

        encoded = tokenizer.encode(req).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(encoded)
            prediction = torch.argmax(output, dim=1).item()

        second_order_flag = tracker.detect_second_order(session_id)

        if prediction == 1 or second_order_flag:
            print("⚠ SQL Injection Detected (Context-Aware)")
        else:
            print("Normal Request")