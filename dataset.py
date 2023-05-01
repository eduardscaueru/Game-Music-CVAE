"""
Preprocesses MIDI files
"""
import numpy as np
import math
import random
from joblib import Parallel, delayed
import multiprocessing

from constants import *
from midi_util import load_midi_v2
from util import *


def compute_beat(beat, notes_in_bar):
    return one_hot(beat % notes_in_bar, notes_in_bar)


def compute_completion(beat, len_melody):
    return np.array([beat / len_melody])


def compute_genre(genre_id):
    """ Computes a vector that represents a particular genre """
    genre_hot = np.zeros((NUM_STYLES,))
    start_index = sum(len(s) for i, s in enumerate(styles) if i < genre_id)
    styles_in_genre = len(styles[genre_id])
    genre_hot[start_index:start_index + styles_in_genre] = 1 / styles_in_genre
    return genre_hot


def stagger(data, time_steps):
    dataX, dataY = [], []
    # Buffer training for first event
    data = ([np.zeros_like(data[0])] * time_steps) + list(data)

    # Chop a sequence into measures
    for i in range(0, len(data) - time_steps, NOTES_PER_BAR):
        dataX.append(data[i:i + time_steps])
        dataY.append(data[i + 1:(i + time_steps + 1)])
    return dataX, dataY


def load_all(styles, batch_size, time_steps):
    """
    Loads all MIDI files as a piano roll.
    (For Keras)
    """
    note_data = []
    beat_data = []
    style_data = []

    note_target = []

    # TODO: Can speed this up with better parallel loading. Order gaurentee.
    styles = [y for x in styles for y in x]

    for style_id, style in enumerate(styles):
        style_hot = one_hot(style_id, NUM_STYLES)
        # Parallel process all files into a list of music sequences
        seqs = Parallel(n_jobs=multiprocessing.cpu_count(), backend='threading')(delayed(load_midi_v2)(f) for f in get_all_files([style]))

        for seq in seqs:
            pm_beats, seq = seq
            if len(seq) >= time_steps:
                # Clamp MIDI to note range
                seq = clamp_midi(seq)
                # Create training data and labels
                train_data, label_data = stagger(seq, time_steps)
                note_data += train_data
                note_target += label_data

                beats = [compute_beat(i, NOTES_PER_BAR) for i in range(len(seq))]
                beat_data += stagger(beats, time_steps)[0]
                # print(np.asarray(beats).shape)

                style_data += stagger([style_hot for i in range(len(seq))], time_steps)[0]

    note_data = np.array(note_data)
    beat_data = np.array(beat_data)
    style_data = np.array(style_data)
    note_target = np.array(note_target)
    return [note_data, note_target, beat_data, style_data], [note_target]


def clamp_midi(sequence):
    """
    Clamps the midi base on the MIN and MAX notes
    """
    new_seq = np.zeros((sequence.shape[0], NUM_NOTES_INSTRUMENT * (NUM_INSTRUMENTS + 1), sequence.shape[2]))
    # print(new_seq.shape)
    for i in range(NUM_INSTRUMENTS + 1):
        # print(i, i * diff, (i + 1) * diff, MIDI_MAX_NOTES * i + MIN_NOTE, MIDI_MAX_NOTES * i + MAX_NOTE)
        new_seq[:, i * NUM_NOTES_INSTRUMENT:(i + 1) * NUM_NOTES_INSTRUMENT, :] = sequence[:, MIDI_MAX_NOTES * i + MIN_NOTE:MIDI_MAX_NOTES * i + MAX_NOTE, :]
    # print(new_seq.shape)
    return new_seq


def unclamp_midi(sequence):
    """
    Restore clamped MIDI sequence back to MIDI note values
    """
    return np.pad(sequence, ((0, 0), (MIN_NOTE, 0), (0, 0)), 'constant')


if __name__ == "__main__":
    data = load_all(styles, BATCH_SIZE, SEQ_LEN)
    print(data[0][3][1, 60, :])
    print(data[0][3].shape)
    print(data[0][0].shape)

    print(np.arange(0, 24 - 24 % BATCH_SIZE, BATCH_SIZE))
    # piece = pm.PrettyMIDI("out/test_in.mid")
    # beats, decoded = midi_decode_v2(piece)
    # print(beats[1], decoded.shape)
    # clamp_midi(decoded)
