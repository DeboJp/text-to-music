import torch
from model import TextToMusic
from utils import generate_midi
import json
import numpy as np

# Load vocabulary
with open("../data/vocab.json", "r") as f:
    vocab = json.load(f)

with open("../data/training_params.json", "r") as f:
    params = json.load(f)

#model init
model = TextToMusic(
    vocab_size=len(vocab),
    embed_dim=params["embed_dim"],
    hidden_dim=params["hidden_dim"],
    output_dim=params["max_note_len"]
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