from datetime import datetime
import json
import os
import keras
import numpy as np
import music21 as m21

MAPPING_PATH = 'mapping.json'

def prepare_seed(seed, mappings, sequence_length):
    seed = seed.split()
    seed = ["/"] * sequence_length + seed
    seed = [mappings[symbol] for symbol in seed]
    return seed

def sample_with_temperature(probabilities, temperature):
    predictions = np.log(probabilities) / temperature
    probabilities = np.exp(predictions) / np.sum(np.exp(predictions))
    
    choices = range(len(probabilities))
    index = np.random.choice(choices, p=probabilities)
    
    return index    

def save_melody_without_midi(melody, step_duration=0.25):
    # Create folder "results" unless exist
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # Name the file has timestamp to avoid duplication
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"generated_{timestamp}_melody_monotone.midi"
    file_path = os.path.join(results_dir, file_name)
    
    # processing
    stream = m21.stream.Stream()
    start_symbol = None
    step_counter = 1

    for i, symbol in enumerate(melody):
        if symbol != "_" or i + 1 == len(melody):
            if start_symbol is not None:
                quarter_length_duration = step_duration * step_counter
                if start_symbol == "r":
                    m21_event = m21.note.Rest(quarterLength=quarter_length_duration)
                else:
                    m21_event = m21.note.Note(int(start_symbol), quarterLength=quarter_length_duration)
                stream.append(m21_event)
                step_counter = 1
            start_symbol = symbol
        else:
            step_counter += 1

    # Save Midi file
    stream.write("midi", file_path)
    print(f"Melody saved to {file_path}")

    # Replace '\' to '/'
    file_url = file_path.replace(os.sep, "/")

    return file_name, file_url

def encode_song(song, time_step=0.25):
    encoded_song = []
    for event in song.flatten().notesAndRests:
        if isinstance(event, m21.note.Note):
            symbol = str(event.pitch.midi)
        elif isinstance(event, m21.note.Rest):
            symbol = "r"
        else:
            continue 
        steps = max(1, int(event.duration.quarterLength / time_step))
        for step in range(steps):
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")
    return " ".join(encoded_song)



def save_melody_with_midi(melody, step_duration=0.25):
    
    # Create folder "results" unless exist
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # Name the file has timestamp to avoid duplication
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"generated_{timestamp}_melody_monotone.midi"
    file_path = os.path.join(results_dir, file_name)

    # processing
    stream = m21.stream.Stream()
    start_symbol = None
    step_counter = 1

    for i, symbol in enumerate(melody):
        if symbol != "_" or i + 1 == len(melody):
            if start_symbol is not None:
                quarter_length_duration = step_duration * step_counter
                if start_symbol == "r":
                    m21_event = m21.note.Rest(quarterLength=quarter_length_duration)
                else:
                    m21_event = m21.note.Note(int(start_symbol), quarterLength=quarter_length_duration)
                stream.append(m21_event)
                step_counter = 1
            start_symbol = symbol
        else:
            step_counter += 1

    # Save MIDI file
    stream.write("midi", file_path)
    print(f"Melody saved to {file_path}")

    return file_name, file_path


def load_model_and_mapping(model_path="model_LSTM.keras"):
    custom_objects = {
        'time_major': False,
        'learning_phase': 0
    }
    model = keras.models.load_model(
        model_path,
        custom_objects=custom_objects
    )
    
    with open(MAPPING_PATH, "r") as fp:
        mappings = json.load(fp)
        
    print("load model success ")
    
    return model, mappings

def generate_melody(model, mappings, seed, num_steps, max_sequence_length, temperature, sequence_length):
    
    start_symbols = ["/"] * sequence_length
    seed = seed.split()
    melody = seed
    seed = start_symbols + seed

    seed = [mappings[symbol] for symbol in seed]
    
    for _ in range(num_steps):
        seed = seed[-max_sequence_length:]
        onehot_seed = keras.utils.to_categorical(seed, num_classes=len(mappings))
        onehot_seed = onehot_seed[np.newaxis, ...]
        
        probabilities = model.predict(onehot_seed)[0]
        output_int = sample_with_temperature(probabilities, temperature)
        
        seed.append(output_int)
        
        output_symbol = [k for k, v in mappings.items() if v == output_int][0]
        
        if output_symbol == "/":
            break
        
        melody.append(output_symbol)
    
    return melody

if __name__ == "__main__":
    model, mappings = load_model_and_mapping()

    # Generate melody without MIDI file input (using seed)
    seed = "67 _ 67 _ 67 _ _ 65 64 _ 64 _ 64 _ _"

    melody = generate_melody(model, mappings, seed, num_steps=1500, max_sequence_length=5000, temperature=0.5, sequence_length=64)

    file_name, file_path = save_melody_without_midi(melody)
    print(melody)

    # Generate melody with MIDI file input
    # file_path = "inputSample/input1.midi"

    # encode midi
    # midi_file = m21.converter.parse(file_path)
    # encoded_seed = encode_song(midi_file)
    
    # print(encoded_seed)

    # melody = generate_melody(model, mappings, encoded_seed, num_steps=1500, max_sequence_length=5000, temperature=0.5, sequence_length=64)

    # file_name, file_path = save_melody_with_midi(melody)