import random
import csv

def generate_notes_sequence(duration_seconds, tempo, base_notes, variation_range = 5):
    notes = []
    beats_per_second = tempo / 60.0
    num_notes = int(duration_seconds * beats_per_second)
    time_per_note = 1.0/ beats_per_second

    for _ in range(num_notes):
        note_variation = random.choice(range(-variation_range,variation_range+1))
        duration = time_per_note * random.choice([0.5, 1, 1.5,2])
        notes.append( (random.choice(base_notes) + note_variation, duration ) )
    return notes

# Base notes for different moods/styles (starting around middle C)
mood_base_notes = {
    "calm": [60, 62, 64, 65, 67],
    "happy": [67, 69, 71, 72, 74],
    "sad": [55, 57, 59, 60, 62],
    "energetic": [72, 74, 76, 77, 79],
    "jazzy": [60, 64, 67, 70, 74],
    "classical": [60, 62, 64, 65, 67, 72, 74],
    "ambient": [50, 55, 60, 62],
    "rock": [50, 55, 57, 60, 62],
    "pop": [64, 67, 69, 71, 72, 76],
    "dance": [62, 67, 71, 74, 78],
    "lullaby": [60, 62, 64, 67]
}


mood_descriptions = {
     "calm": [
        "calm and soothing melody, slow tempo",
        "soft, serene, relaxing tune, gentle pace",
        "peaceful and tranquil ambient music, slow and quiet",
         "meditative slow moving sound",
        "gentle flowing music, light and airy"
    ],
    "happy": [
        "bright, cheerful, upbeat tune, moderate tempo",
        "joyful and lively rhythm, energetic",
        "bouncy and playful melody, fast rhythm",
         "upbeat and energetic sound",
        "happy light hearted tune"
    ],
    "sad": [
         "melancholic slow melody, somber and reflective",
         "heartfelt, sorrowful and slow rhythm, emotional",
         "downbeat, low sound",
         "pensive and reflective tune",
        "sad and slow paced melody"
    ],
    "energetic": [
        "fast-paced, lively and high energy rhythm",
         "driving and intense beat, upbeat and fast",
         "powerful and fast paced music",
         "thrilling fast energetic rhythm",
        "dynamic and fast tempo beat"
    ],
     "jazzy": [
         "swinging jazzy rhythm, upbeat",
        "smooth and syncopated jazz",
        "complex and improvisational jazzy melody",
        "cool and mellow jazzy sounds",
        "relaxed and groovy jazz"
    ],
    "classical": [
        "elegant and refined classical piece",
        "majestic and orchestral classical tune",
         "structured and detailed classical music",
         "formal and refined sound",
        "graceful and balanced classical piece"
    ],
    "ambient": [
        "atmospheric and ethereal ambient sound",
         "dreamy and spacious ambient music",
        "slow and evolving ambient textures",
        "abstract and peaceful soundscape",
        "gentle and floating ambient background"
    ],
     "rock": [
        "heavy and distorted rock riff",
        "loud and powerful rock beat",
         "energetic and driving rock rhythm",
         "intense and distorted sound",
        "raw and powerful rock energy"
    ],
    "pop": [
        "catchy and commercial pop melody",
         "radio-friendly upbeat pop rhythm",
        "modern and mainstream pop",
         "easy listening pop tune",
         "accessible and popular sound"
    ],
    "dance": [
        "high-energy dance beat, fast tempo",
        "pulsating and rhythmic dance",
        "driving and intense dance sound",
        "upbeat and groovy dance rhythm",
        "electronic and vibrant dance music"
    ],
     "lullaby": [
         "soft and gentle lullaby, slow tempo",
         "calming and soothing lullaby",
         "peaceful and quiet lullaby",
        "relaxing bedtime music",
        "tender and sweet lullaby"
    ]

}


# Generate dataset
dataset = []
for mood, base_notes in mood_base_notes.items():
    for desc in mood_descriptions[mood]:
        tempo = random.choice([60, 80, 100, 120,140, 160])
        duration = 20 #seconds
        note_sequence = generate_notes_sequence(duration, tempo, base_notes)
        
        # Format notes as a string of note and durations for csv compatibility
        formatted_notes = " ".join(f"{note}:{duration}" for note, duration in note_sequence)

        dataset.append((desc, formatted_notes))

# Write to CSV
with open("dataset.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["text", "notes"])
    writer.writerows(dataset)

print("Dataset generated successfully!")