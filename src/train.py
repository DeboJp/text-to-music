import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from model import TextToMusic
import json

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
    return [vocab.get(word, vocab["<PAD>"]) for word in text.split()]

def notes_to_tensor(notes_str):
  
  notes = []
  durations = []
  
  for note_duration in notes_str.split():
      note,duration = note_duration.split(":")
      notes.append(int(note))
      durations.append(float(duration))
  return torch.tensor(notes), torch.tensor(durations)
  

inputs = [text_to_sequence(text) for text in df["text"]]
notes_and_durations = [notes_to_tensor(notes) for notes in df["notes"]]


# Pad sequences
max_len = max(len(seq) for seq in inputs)
inputs = torch.tensor([seq + [vocab["<PAD>"]] * (max_len - len(seq)) for seq in inputs])

#create targets
max_note_len = max(len(notes) for notes,duration in notes_and_durations )
padded_notes = []
padded_durations = []
for notes, durations in notes_and_durations:
  
    padded_notes.append(torch.cat( [notes , torch.zeros(max_note_len - len(notes)) ] , dim = 0 ))
    padded_durations.append(torch.cat([durations, torch.zeros(max_note_len - len(durations)) ], dim = 0))

targets_notes = torch.stack(padded_notes)
targets_durations = torch.stack(padded_durations)



# Initialize model
model = TextToMusic(
    vocab_size=len(vocab),
    embed_dim=16,
    hidden_dim=32,
    output_dim=max_note_len
)

criterion_notes = nn.MSELoss()
criterion_durations = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train
for epoch in range(100):
    optimizer.zero_grad()
    predicted_notes,predicted_durations = model(inputs)
    loss_notes = criterion_notes(predicted_notes, targets_notes.float())
    loss_durations = criterion_durations(predicted_durations, targets_durations.float())
    loss = loss_notes + loss_durations
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Save model
import os

torch.save(model.state_dict(), "../model.pth")
print("Saving model to:", os.path.abspath("../model.pth"))

# Save vocabulary
with open("../data/vocab.json", "w") as f:
    json.dump(vocab, f)