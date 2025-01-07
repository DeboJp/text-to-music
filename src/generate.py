import torch
from model import TextToMusic
from utils import generate_midi
import json

# Load vocabulary
with open("../data/vocab.json", "r") as f:
    vocab = json.load(f)

# Model initialization
model = TextToMusic(
    vocab_size=len(vocab),
    embed_dim=16,
    hidden_dim=32,
    output_dim=5
)
model.load_state_dict(torch.load("../model.pth"))
model.eval()

# Convert text to sequence
def text_to_sequence(text):
    return [vocab.get(word, vocab["<PAD>"]) for word in text.split()]

# Generate music
text_input = "calm and soothing melody"
sequence = torch.tensor([text_to_sequence(text_input)])
predicted_notes = model(sequence).detach().numpy().flatten()
generate_midi(predicted_notes, "generated_music.mid")
