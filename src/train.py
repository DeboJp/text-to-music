import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from model import TextToMusic

# Load dataset
df = pd.read_csv("../data/dataset.csv")

# Build vocabulary dynamically
vocab = {"<PAD>": 0}  # Add a special token for padding
for text in df["text"]:
    for word in text.split():
        if word not in vocab:
            vocab[word] = len(vocab)

print("Vocabulary:", vocab)  # Optional: Print to verify vocabulary

# Preprocess data
def text_to_sequence(text):
    return [vocab[word] for word in text.split()]

inputs = [text_to_sequence(text) for text in df["text"]]
targets = [[int(note) for note in notes.split()] for notes in df["notes"]]

# Pad sequences
max_len = max(len(seq) for seq in inputs)
inputs = torch.tensor([seq + [vocab["<PAD>"]] * (max_len - len(seq)) for seq in inputs])
targets = torch.tensor(targets)

# Initialize model
model = TextToMusic(
    vocab_size=len(vocab),
    embed_dim=16,
    hidden_dim=32,
    output_dim=targets.shape[1]
)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets.float())
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Save model
torch.save(model.state_dict(), "../model.pth")

import json

# Save vocabulary
with open("../data/vocab.json", "w") as f:
    json.dump(vocab, f)
