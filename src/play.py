import pygame

# Initialize the mixer
pygame.mixer.init()

# Load and play the MIDI file
pygame.mixer.music.load("./generated_music.mid")
pygame.mixer.music.play()

# Wait until the music finishes
while pygame.mixer.music.get_busy():
    pass
