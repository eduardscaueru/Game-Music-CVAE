# from train import train_model
from dataset import load_all
from constants import *
import numpy as np
import tensorflow as tf
import pretty_midi as pm
from midi_util import midi_encode_v2, idx_to_instrument


def generate_song(cvae, length, style_label):
    generated_seqs = np.zeros((length, cvae.decoder_block.output.shape[1], cvae.decoder_block.output.shape[2]))

    for i in range(length):
        z = tf.random.normal(shape=(1, cvae.latent_block.output[0].shape[1]), mean=0.0, stddev=1.0)
        z_lbl_concat = np.concatenate((z, style_label), axis=1)
        predicted_seq = cvae.decoder_block(z_lbl_concat)

        generated_seqs[i, :, :] = predicted_seq

    return generated_seqs


if __name__ == "__main__":
    # pass
    data = load_all(styles, BATCH_SIZE, SEQ_LEN)
    # model, _, _ = train_model(LATENT_DIM, EPOCHS, data)
    model_name = 'test'
    model = tf.keras.models.load_model('out/models/' + model_name)
    model.summary()
    label = np.zeros((1, NUM_STYLES))
    label[:, 0] = 1
    generated = generate_song(model, 5, label)
    print(np.max(generated))
    print(len(generated[generated > 0.1]))
    print(generated.shape)

    t = 0
    final = np.zeros((NUM_INSTRUMENTS + 1, generated.shape[0] * generated.shape[1], NUM_NOTES_INSTRUMENT))
    instrument_max_probs = {i: 0 for i in range(NUM_INSTRUMENTS + 1)}
    print(final.shape)
    for bars in range(generated.shape[0]):
        for time_step in range(generated.shape[1]):
            for i in range(NUM_INSTRUMENTS + 1):
                instrument_seq = generated[bars, time_step, i * NUM_NOTES_INSTRUMENT:(i + 1) * NUM_NOTES_INSTRUMENT]
                selected_note_idx = np.argmax(instrument_seq)
                max_prob = np.max(instrument_seq)

                instrument_max_probs[i] += max_prob

                final[i, t, selected_note_idx] = 1
            t += 1

    final.tofile('out/generated.dat')

    sorted_instruments = sorted(instrument_max_probs.items(), key=lambda x: x[1], reverse=True)
    print(sorted_instruments)
    selected_instruments = [(idx_to_instrument[x[0]], final[x[0], :, :]) for x in sorted_instruments] #sorted_instruments[:4]

    # selected_instruments = []
    # for instrument_idx in range(NUM_INSTRUMENTS + 1):
    #     if np.sum(final[instrument_idx, :, :, 1]) > 0:
    #         print(instrument_idx, idx_to_instrument[instrument_idx])
    #         selected_instruments.append((idx_to_instrument[instrument_idx], final[instrument_idx, :, :, :]))

    pm_song = pm.PrettyMIDI()
    for program, piano_roll in selected_instruments:
        encoded = midi_encode_v2(piano_roll, program=program)
        pm_song.instruments.append(encoded.instruments[0])

    f = open("out/generated_test_loaded_model.mid", "w")
    f.close()
    pm_song.write("out/generated_test_loaded_model.mid")

