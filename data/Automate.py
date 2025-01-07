import random

moods = {
    "calm": [60, 62, 64, 65, 67],
    "happy": [67, 69, 71, 72, 74],
    "sad": [55, 57, 59, 60, 62],
    "energetic": [72, 74, 76, 77, 79]
}

descriptions = {
    "calm": ["calm soothing melody", "soft serene tune"],
    "happy": ["bright cheerful tune", "upbeat joyful rhythm"],
    "sad": ["melancholic slow melody", "somber reflective tune"],
    "energetic": ["fast-paced lively rhythm", "energetic driving beat"]
}

# Generate dataset
dataset = []
for mood, notes in moods.items():
    for desc in descriptions[mood]:
        # Add random variations
        variation = [note + random.choice([-2, 0, 2]) for note in notes]
        dataset.append((desc, " ".join(map(str, variation))))

# Write to CSV
import csv
with open("dataset.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["text", "notes"])
    writer.writerows(dataset)
