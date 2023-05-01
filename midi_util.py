"""
Handles MIDI file loading
"""
import pretty_midi as pm
import numpy as np
from constants import *
from copy import deepcopy
import os
import glob
import tensorflow as tf
from more_itertools.more import unzip


def load_midi_v2(fname):

    print(fname)
    p = pm.PrettyMIDI(fname)
    cache_path = os.path.join(CACHE_DIR, fname + '.npy')
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    note_seq = midi_decode_v2(p)

    return note_seq


def compute_drum_piano_roll(instrument, piano_roll, fs):

    roll = deepcopy(piano_roll)
    for note in instrument.notes:
        # Should interpolate
        roll[note.pitch, int(note.start * fs):int(note.end * fs)] += note.velocity

    return roll


def midi_decode_v2(p):

    # Compute piano rolls for every instrument
    # Remove duplicated instruments and keep only the one with max notes length
    instruments = {}
    sorted_instruments = sorted(p.instruments, key=lambda x: len(x.notes), reverse=True)
    for instrument in sorted_instruments:
        if instrument.program not in instruments:
            instruments[instrument.program] = instrument
    # for instrument in instruments.values():
    #     print(instrument, pretty_midi.program_to_instrument_name(instrument.program), len(instrument.notes))
    # TODO: compute for drums separately because pretty midi can't
    # TODO: determine the frequency for a 16th note? (but if the fs is higher then no replay matrix is needed since the
    #   pauses between consecutive notes are captured)
    piano_rolls = [] # [[instrument, t_play, t_volume]]
    for instrument in instruments.values():
        # print(pm.program_to_instrument_name(instrument.program), instrument.is_drum,
        #       instrument.get_piano_roll(FS).shape)
        if instrument.is_drum:
            piano_rolls.append([instrument, compute_drum_piano_roll(instrument, instrument.get_piano_roll(FS), FS)])
        else:
            piano_rolls.append([instrument, instrument.get_piano_roll(FS)])

    # Pad the smaller piano rolls with zeros so that all instruments have the same time_steps
    pitches, max_time_steps = sorted([piano_roll[1].shape for piano_roll in piano_rolls], key=lambda x: -x[1])[0]
    for piano_roll in piano_rolls:
        padding = np.zeros((pitches, max_time_steps - piano_roll[1].shape[1]))
        piano_roll[1] = np.concatenate((piano_roll[1], padding), axis=1)

        # Compute the 'play' and 'velocity' matrices
        # TODO: should I leave the piano roll as it is right now with the velocity in it?
        t_volume = deepcopy(piano_roll[1])
        normalize_velocity = lambda v: v / MAX_VELOCITY
        vfunc = np.vectorize(normalize_velocity)
        t_volume = vfunc(t_volume)

        t_play = piano_roll[1]
        t_play[t_play == 100] = 1

        piano_roll.append(t_volume)

    # Compute final array with instrument dimension
    # TODO: what should be the instrument encoding for drums? for now in NUM_INSTRUMENTS
    # drum_roll_play = np.zeros((pitches, max_time_steps))
    # drum_roll_volume = np.zeros((pitches, max_time_steps))
    final = np.zeros((max_time_steps, (NUM_INSTRUMENTS + 1) * pitches, 2)) # + drum dimension
    drums_sorted = sorted([len(instrument.notes) for instrument in p.instruments if instrument.is_drum], reverse=True)
    max_drum_notes = None
    if len(drums_sorted) > 0:
        max_drum_notes = drums_sorted[0]
    for piano_roll in piano_rolls:
        instrument = piano_roll[0]
        t_play = piano_roll[1]
        t_volume = piano_roll[2]

        # If there are more drums then get only the one with the most notes.
        if instrument.is_drum and len(instrument.notes) == max_drum_notes:
            # drum_roll_play = drum_roll_play + t_play
            # drum_roll_volume = drum_roll_volume + t_volume
            instrument_idx = instrument_to_idx[-1]
            final[:, instrument_idx * pitches:(instrument_idx + 1) * pitches, 0] = t_play.T
            final[:, instrument_idx * pitches:(instrument_idx + 1) * pitches, 1] = t_volume.T
        elif instrument.program in instrument_to_idx:
            # print(np.stack([t_volume.T, t_play.T], axis=2).shape)
            # print(t_volume.flatten('F')[20*128:21*128])
            # print(instrument.program)
            # print(t_volume.shape) # (128, 383)
            # print(t_volume.flatten('F').shape)
            # print(t_volume[:, 20])
            # print(final[:, :, 0].shape)
            # print(final[:, instrument.program * pitches:(instrument.program + 1) * pitches, 0].shape) # (383, 128)
            instrument_idx = instrument_to_idx[instrument.program]
            final[:, instrument_idx * pitches:(instrument_idx + 1) * pitches, 0] = t_play.T
            final[:, instrument_idx * pitches:(instrument_idx + 1) * pitches, 1] = t_volume.T
            #final[:, instrument.program] = np.stack([t_volume.T, t_play.T], axis=2) # 37

    # Limit the notes in any case there are more drums
    # drum_roll_play[drum_roll_play > 1] = 1
    # drum_roll_volume[drum_roll_volume > 1] = 1
    # final[:, NUM_INSTRUMENTS * pitches:(NUM_INSTRUMENTS + 1) * pitches, 0] = drum_roll_play.T
    # final[:, NUM_INSTRUMENTS * pitches:(NUM_INSTRUMENTS + 1) * pitches, 1] = drum_roll_volume.T
    # print(final[20, 37*pitches:38*pitches, 0])
    # drum_roll = np.stack([drum_roll_volume.T, drum_roll_play.T], axis=2)
    # final[:, NUM_INSTRUMENTS] = drum_roll
    #print(final[NUM_INSTRUMENTS, :, max_time_steps - 100, 1])

    beat_duration = p.get_beats()[1] - p.get_beats()[0]

    return p.get_beats(), final


def transpose_keys(filename, out_dir):

    one_track_midi = pm.PrettyMIDI(filename)
    tempo = pm.get_tempo_changes()[1][0]
    for i in range(12):
        midi_transposed = pm.PrettyMIDI(initial_tempo=tempo)
        midi_transposed.time_signature_changes = deepcopy(one_track_midi.time_signature_changes)
        midi_transposed.key_signature_changes = [pm.KeySignature(i, 0)]
        midi_transposed.instruments = deepcopy(one_track_midi.instruments)

        file_name = out_dir + "/" + filename.split("/")[-1][:-4] + "_" + pm.key_number_to_key_name(i).replace(" ", "_") + ".mid"
        print(file_name)
        f = open(file_name, "w")
        f.close()
        midi_transposed.write(file_name)


def are_same_notes(f1, f2):

    p1 = pm.PrettyMIDI(f1)
    p2 = pm.PrettyMIDI(f2)

    for instrument_idx in range(len(p1.instruments)):
        for i in range(len(p1.instruments[instrument_idx].notes)):
            if p1.instruments[instrument_idx].notes[i].pitch != p2.instruments[instrument_idx].notes[i].pitch:
                return False
    return True


def delete_same_files(dir_name):

    transposed_files = glob.glob(dir_name + "/*.mid")
    to_delete = {}

    for t1 in transposed_files:
        for t2 in transposed_files:
            if t1 != t2:
                # Check if they are the same
                if are_same_notes(t1, t2):
                    if t1 not in to_delete:
                        to_delete[t1] = [t2]
                    else:
                        to_delete[t1].append(t2)

    for key, val in to_delete.items():
        if glob.glob(key):
            for file in val:
                os.remove(file)


def limit_instruments():
    instrument_freq = {}
    for root, dirs, files in os.walk("./data", topdown=False):
        for name in files:
            midi_file_name = os.path.join(root, name)
            song = pm.PrettyMIDI(midi_file_name)
            sorted_instruments_by_notes = sorted([(len(x.notes), x.program) for x in song.instruments],
                                                 key=lambda x: x[0], reverse=True)[:MAX_INSTRUMENTS_PER_SONG]
            _, programs = unzip(sorted_instruments_by_notes)
            for program in programs:
                if program not in instrument_freq:
                    instrument_freq[program] = 1

    # print(instrument_freq)
    # print(sorted(list(instrument_freq.items()), key=lambda x: x[1], reverse=True))
    programs = list(instrument_freq.keys())
    program_to_idx = list(zip(programs, np.arange(len(programs))))
    program_to_idx.append((-1, len(program_to_idx)))
    # print(dict(program_to_idx))
    print("NUM_INSTRUMENTS={}".format(len(program_to_idx) - 1))

    return dict(program_to_idx)


instrument_to_idx = limit_instruments()


if __name__ == '__main__':
    # Test
    # pp = mido.MidiFile("out/test_in.mid")
    # midi_decode_v1(pp)
    print("INTRUMENT to IDX:", instrument_to_idx)
    piece = pm.PrettyMIDI("out/test_in.mid")
    beats, decoded = midi_decode_v2(piece)
    print(beats[1], decoded.shape)
    # print(decoded[32, 43 * 128:(43 + 1) * 128, 1])
    #p = midi_encode(midi_decode(p))
    #midi.write_midifile("out/test_out.mid", p)

    # # pitch_class_matrix = np.array([one_hot(n % OCTAVE, OCTAVE) for n in range(NUM_NOTES_INSTRUMENT)])  # notes, octaves
    # pitch_class_matrix = np.array([np.tile(one_hot(n % OCTAVE, OCTAVE), NUM_INSTRUMENTS + 1) for n in range(NUM_NOTES_INSTRUMENT)])  # notes, octaves
    # pitch_class_matrix = tf.constant(pitch_class_matrix, dtype='float32')
    # print(pitch_class_matrix)
    # pitch_class_matrix = tf.reshape(pitch_class_matrix, [1, 1, NUM_NOTES, OCTAVE])
    # print(pitch_class_matrix)
    #
    # decoded = np.asarray([decoded])
    # # decoded = decoded[:, :16*5, :128, :]
    # print(decoded.shape)
    # octaves_t = []
    # for i in range(OCTAVE):
    #     d = decoded[:, :, i::OCTAVE, 0]
    #     # print(d.shape[-1] % OCTAVE)
    #     if d.shape[-1] % (OCTAVE - 1) != 0:
    #         # print(np.zeros((decoded.shape[0], decoded.shape[1], (OCTAVE - 1) - decoded[:, :, i::OCTAVE, 0].shape[-1])).shape)
    #         # print(i, OCTAVE, "padded", np.append(d, np.zeros((decoded.shape[0], decoded.shape[1], (OCTAVE - 1) - (d.shape[-1] % OCTAVE))), axis=2).shape)
    #         octaves_t.append(np.append(d, np.zeros((d.shape[0], d.shape[1], (OCTAVE - 1) - (d.shape[-1] % OCTAVE))), axis=2))
    #     else:
    #         # print(i ,OCTAVE, decoded[:, :, i::OCTAVE, 0].shape)
    #         octaves_t.append(d)
    #
    # # print(tf.constant(np.asarray([decoded[:, i::OCTAVE, 0] for i in range(OCTAVE)]), dtype='float32'))
    # print("octaves", np.asarray(octaves_t).shape)
    # bins_t = tf.reduce_sum(octaves_t, axis=3)
    # print(bins_t.shape)
    # bins_t = tf.tile(bins_t, [NUM_NOTES // OCTAVE, 1, 1])
    # print(bins_t.shape)
    # bins_t = tf.reshape(bins_t, [decoded.shape[0], decoded.shape[1], NUM_NOTES, 1])
    # print(bins_t.shape)

