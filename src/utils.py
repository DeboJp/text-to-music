import pretty_midi

def generate_midi(predicted_notes, output_file="output.mid"):
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano
    start_time = 0
    for note,duration in predicted_notes:
        note_obj = pretty_midi.Note(velocity=100, pitch=int(note), start=start_time, end=start_time + duration)
        piano.notes.append(note_obj)
        start_time += duration
    midi.instruments.append(piano)
    midi.write(output_file)