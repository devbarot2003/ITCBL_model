import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.ITCBLModel import ITCBL
from preprocessing.prepare_data import tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = ITCBL(vocab_size=10000).to(device)
model.load_state_dict(torch.load("models/sqli_model.pth", map_location=device))
model.eval()


def explain_query(query):

    tokens = query.split()

    print(f"\nQuery: {query}")
    print("\nToken Importance:\n")

    base_input = tokenizer.encode(query).unsqueeze(0).to(device)

    with torch.no_grad():
        base_output = model(base_input)
        base_score = torch.softmax(base_output, dim=1)[0][1].item()

    token_scores = []

    for i, token in enumerate(tokens):

        modified_tokens = tokens.copy()
        modified_tokens[i] = ""   # remove token

        modified_query = " ".join(modified_tokens)

        modified_input = tokenizer.encode(modified_query).unsqueeze(0).to(device)

        with torch.no_grad():
            modified_output = model(modified_input)
            modified_score = torch.softmax(modified_output, dim=1)[0][1].item()

        importance = base_score - modified_score

        token_scores.append((token, importance))

    # Sort by importance
    token_scores = sorted(token_scores, key=lambda x: x[1], reverse=True)

    for token, score in token_scores:
        print(f"{token} → {score:.4f}")


# -----------------------------
# TEST EXPLANATION
# -----------------------------
test_query = "GET /login?id=1 OR 1=1 --"

explain_query(test_query)