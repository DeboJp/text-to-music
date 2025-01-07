import torch
from model import TextToMusic
from utils import generate_midi
import json
import numpy as np

# Load vocabulary
with open("../data/vocab.json", "r") as f:
    vocab = json.load(f)

# Model initialization
model = TextToMusic(
    vocab_size=len(vocab),
    embed_dim=16,
    hidden_dim=32,
    output_dim=200
)

import os

print("Loading model from:", os.path.abspath("../model.pth"))
model.load_state_dict(torch.load("../model.pth"))

model.eval()

# Convert text to sequence
def text_to_sequence(text):
    return [vocab.get(word, vocab["<PAD>"]) for word in text.split()]

# Generate music
text_input = "calm and soothing melody"
sequence = torch.tensor([text_to_sequence(text_input)])

predicted_notes, predicted_durations = model(sequence)
predicted_notes = predicted_notes.detach().numpy().flatten()
predicted_durations = predicted_durations.detach().numpy().flatten()

# Convert to list of tuples
predicted_notes_with_durations = list(zip(predicted_notes,predicted_durations))
generate_midi(predicted_notes_with_durations, "generated_music.mid")