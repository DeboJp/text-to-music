import torch
import torch.nn as nn

class TextToMusic(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(TextToMusic, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc_notes = nn.Linear(hidden_dim, output_dim)
        self.fc_durations = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        notes = self.fc_notes(out[:, -1, :])
        durations = self.fc_durations(out[:, -1, :])

        return notes, durations