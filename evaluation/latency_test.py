import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import time
from models.ITCBLModel import ITCBL
from preprocessing.prepare_data import test_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ITCBL(vocab_size=10000).to(device)
model.load_state_dict(torch.load("models/sqli_model.pth"))
model.eval()

total_time = 0
total_samples = 0

with torch.no_grad():
    for inputs, _ in test_loader:
        inputs = inputs.to(device)

        start = time.time()

        outputs = model(inputs)

        end = time.time()

        total_time += (end - start)
        total_samples += inputs.size(0)

avg_latency = total_time / total_samples

print("Average inference time per request:", avg_latency*1000, "milliseconds")