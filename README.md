Instructions for running the project.

# Text-to-Music

A lightweight prototype that generates MIDI music from text descriptions.

## How to Run

1. **Install Dependencies**:
pip install -r requirements.txt


2. **Train the Model**:
python src/train.py

3. **Generate Music**:
python src/generate.py


4. **Output**:
The generated MIDI file will be saved as `generated_music.mid` in the `data/` directory.

# How to Use
Place your text-MIDI dataset in data/dataset.csv.
Run train.py to train the model.
Use generate.py to generate MIDI files from text input.
Play the MIDI files using any player (e.g., MuseScore).
