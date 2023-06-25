# from train import train_model
from dataset import load_all, unclamp_midi
from constants import *
import numpy as np
import tensorflow as tf
import pretty_midi as pm
from midi_util import midi_encode_v2, limit_instruments
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.utils import shuffle
from model import sampling
from keras.layers import RepeatVector


def plot_latent_space_per_genre(cvae, instrument_to_idx, genre_idx):
    dataset = load_all(styles, SEQ_LEN, instrument_to_idx)
    note_data = dataset[0][0]
    style_data = dataset[0][3]
    game_data = dataset[0][4]
    num_seqs = note_data.shape[0]
    note_data, style_data, game_data = shuffle(note_data, style_data, game_data, random_state=42)
    z_mu_label = []
    labels = []

    for start_batch in np.arange(0, num_seqs - num_seqs % BATCH_SIZE, BATCH_SIZE):
        seqs = note_data[start_batch:start_batch + BATCH_SIZE, :, :]
        style_labels = style_data[start_batch:start_batch + BATCH_SIZE, 1, :]
        game_labels = game_data[start_batch:start_batch + BATCH_SIZE, 1, :]

        encoder_predict = cvae.encoder_block(seqs)
        style_lbl_concat = tf.concat([encoder_predict, style_labels], 1)
        z_mu, z_rho = cvae.latent_block(style_lbl_concat)

        # print(z_mu.shape)
        for i in range(BATCH_SIZE):
            label = style_labels[i, :]
            label_idx = np.argmax(label)
            # print("genre_label + idx", label, label_idx, genre)
            if label_idx == genre_idx:
                game_label = game_labels[i, :]
                game_idx = np.argmax(game_label)
                # print("game_label + idx", game_label, game_idx)
                labels.append(game_idx)
                z_mu_label.append(z_mu[i, :])

    z_mu_label = np.asarray(z_mu_label)
    # print("beofre pca :", z_mu_label.shape, np.asarray(labels).shape)
    z_mu_pca = TSNE(n_components=2, perplexity=5, n_iter=1000, init="pca").fit_transform(z_mu_label)

    z_mu_final_x = z_mu_pca[:, 0].tolist()
    z_mu_final_y = z_mu_pca[:, 1].tolist()

    plt.figure(figsize=(12, 10))
    plt.scatter(z_mu_final_x, z_mu_final_y, c=labels)
    plt.colorbar()
    plt.xlabel("z_mu_pca[0]")
    plt.ylabel("z_mu_pca[1]")
    plt.title("Latent space for the games in genre " + genre[genre_idx])
    plt.show()


def plot_latent_space(cvae, instrument_to_idx):
    dataset = load_all(styles, SEQ_LEN, instrument_to_idx)
    note_data = dataset[0][0]
    style_data = dataset[0][3]
    num_seqs = note_data.shape[0]
    z_mu_final_x = []
    z_mu_final_y = []
    labels_final = []
    note_data, style_data = shuffle(note_data, style_data, random_state=42)

    for start_batch in np.arange(0, num_seqs - num_seqs % BATCH_SIZE, BATCH_SIZE):
        seqs = note_data[start_batch:start_batch + BATCH_SIZE, :, :]
        style_labels = style_data[start_batch:start_batch + BATCH_SIZE, 1, :]

        encoder_predict = cvae.encoder_block(seqs)
        style_lbl_concat = tf.concat([encoder_predict, style_labels], 1)
        z_mu, z_rho = cvae.latent_block(style_lbl_concat)

        # print(z_mu.shape)
        z_mu_label = []
        labels = []
        idx = []
        for i in range(BATCH_SIZE):
            label = style_labels[i, :]
            game = np.argmax(label)
            labels.append(game)
            idx.append(int(i))
            z_mu_label.append(z_mu[i, :])
        # print(idx)
        # z_mu = tf.gather(z_mu, indices=idx)
        if len(labels) < 5:
            break
        # pca = PCA(n_components=2)
        # z_mu_pca = pca.fit_transform(z_mu)
        z_mu_label = np.asarray(z_mu_label)
        # print("beofre pca :",z_mu_label.shape, np.asarray(labels).shape)
        z_mu_pca = TSNE(n_components=2, perplexity=5, n_iter=1000, init="pca").fit_transform(z_mu_label)

        z_mu_final_x += z_mu_pca[:, 0].tolist()
        z_mu_final_y += z_mu_pca[:, 1].tolist()
        labels_final += labels

    plt.figure(figsize=(12, 10))
    plt.scatter(z_mu_final_x, z_mu_final_y, c=labels_final)
    plt.colorbar()
    plt.xlabel("z_mu_pca[0]")
    plt.ylabel("z_mu_pca[1]")
    plt.show()


def interpolate(cvae, instrument_to_idx, a, b, num_steps):
    dataset = load_all(styles, SEQ_LEN, instrument_to_idx)
    note_data = dataset[0][0]
    style_data = dataset[0][3]
    game_data = dataset[0][4]
    num_seqs = note_data.shape[0]
    note_data, style_data, game_data = shuffle(note_data, style_data, game_data, random_state=42)
    sample_a = None
    sample_b = None
    genre_label = None
    ok = 0

    for start_batch in np.arange(0, num_seqs - num_seqs % BATCH_SIZE, BATCH_SIZE):
        seqs = note_data[start_batch:start_batch + BATCH_SIZE, :, :]
        style_labels = style_data[start_batch:start_batch + BATCH_SIZE, 1, :]
        game_labels = game_data[start_batch:start_batch + BATCH_SIZE, 1, :]

        for i in range(BATCH_SIZE):
            game_label = game_labels[i, :]
            game_idx = np.argmax(game_label)
            if sample_a is None and a == game_idx:
                print("game_idx a", game_idx)
                sample_a = seqs[i, :, :]
                genre_label = style_labels[i, :]
            if sample_b is None and b == game_idx:
                print("game_idx b", game_idx)
                sample_b = seqs[i, :, :]
                genre_label = style_labels[i, :]
            if sample_a is not None and sample_b is not None:
                ok = 1
                break
        if ok == 0:
            break

    # sample_a = tf.transpose(RepeatVector(BATCH_SIZE)(sample_a), perm=[1, 0, 2])
    # sample_b = tf.transpose(RepeatVector(BATCH_SIZE)(sample_b), perm=[1, 0, 2])
    sample_a = tf.expand_dims(sample_a, axis=0)
    sample_b = tf.expand_dims(sample_b, axis=0)
    # genre_label = np.tile(genre_label, (BATCH_SIZE, 1))
    genre_label = tf.expand_dims(genre_label, axis=0)
    print(sample_a.shape, sample_b.shape, genre_label.shape)

    z_a = cvae.encoder_block(sample_a)
    z_b = cvae.encoder_block(sample_b)

    z_a_lbl_concat = tf.concat([z_a, genre_label], 1)
    z_b_lbl_concat = tf.concat([z_b, genre_label], 1)
    z_mu_a, z_rho_a = cvae.latent_block(z_a_lbl_concat)
    z_mu_b, z_rho_b = cvae.latent_block(z_b_lbl_concat)
    z_a = sampling(z_mu_a, z_rho_a, 1, LATENT_DIM)
    z_b = sampling(z_mu_b, z_rho_b, 1, LATENT_DIM)

    z_a = z_a[0, :]
    z_b = z_b[0, :]
    diff = z_b - z_a
    step_size = 1 / num_steps
    steps = tf.range(0, 1 + step_size, step_size)
    interpolations = []
    for step in steps:
        interpolations.append(z_a + step * diff)
    decoded_seqs = []
    for interpolation in interpolations:
        interpolation = tf.expand_dims(interpolation, axis=0)
        z_decoder = tf.concat([interpolation, genre_label], 1)
        decoded_seq = cvae.decoder_block(z_decoder)
        decoded_seqs.append(decoded_seq[0, :, :])

    return decoded_seqs


def decoder_predict(cvae, length, style_label):
    generated_seqs = np.zeros((length, cvae.decoder_block.output.shape[1], cvae.decoder_block.output.shape[2]))

    for i in range(length):
        z = tf.random.normal(shape=(1, cvae.latent_block.output[0].shape[1]), mean=0.0, stddev=1.0)
        z_lbl_concat = np.concatenate((z, style_label), axis=1)
        predicted_seq = cvae.decoder_block(z_lbl_concat)

        generated_seqs[i, :, :] = predicted_seq

    return generated_seqs


def select_note(instrument_seq, strategy="GREEDY"):
    if strategy == "GREEDY":
        return [np.argmax(instrument_seq)]

    softmax_seq = tf.nn.softmax(instrument_seq)
    selected_idx = np.random.choice(instrument_seq.shape[0], 1, p=softmax_seq)

    return selected_idx


def separate_instruments(generated, idx_to_instrument, strategy, threshold=1e-2):
    t = 0
    final = np.zeros(
        (NUM_INSTRUMENTS + 1, generated.shape[0] * generated.shape[1], NUM_NOTES_INSTRUMENT))
    instrument_max_probs = {i: 0 for i in range(NUM_INSTRUMENTS + 1)}
    print(final.shape)
    for bars in range(generated.shape[0]):
        for time_step in range(generated.shape[1]):
            for i in range(NUM_INSTRUMENTS + 1):
                instrument_seq = generated[bars, time_step, i * NUM_NOTES_INSTRUMENT:(i + 1) * NUM_NOTES_INSTRUMENT]
                selected_notes_idx = select_note(instrument_seq, strategy=strategy)
                max_prob = np.max(instrument_seq)
                # print(max_prob)
                if max_prob >= threshold:
                    instrument_max_probs[i] += max_prob

                    for selected_note_idx in selected_notes_idx:
                        final[i, t, selected_note_idx] = 1
            t += 1

    final.tofile('out/generated.dat')

    sorted_instruments = sorted(instrument_max_probs.items(), key=lambda x: x[1], reverse=True)
    print(sorted_instruments)
    selected_instruments = [(idx_to_instrument[x[0]], final[x[0], :, :]) for x in
                            sorted_instruments[:MAX_INSTRUMENTS_GENERATED]]

    return selected_instruments


def generate(cvae, length, label, idx_to_instrument, strategy):
    generated = decoder_predict(cvae, length, label)
    print(np.max(generated))
    print(len(generated[generated > 0.1]))
    print(generated.shape)

    selected_instruments = separate_instruments(generated, idx_to_instrument, strategy)

    # selected_instruments = []
    # for instrument_idx in range(NUM_INSTRUMENTS + 1):
    #     if np.sum(final[instrument_idx, :, :, 1]) > 0:
    #         print(instrument_idx, idx_to_instrument[instrument_idx])
    #         selected_instruments.append((idx_to_instrument[instrument_idx], final[instrument_idx, :, :, :]))

    pm_song = pm.PrettyMIDI()
    for program, piano_roll in selected_instruments:
        print(program)
        unclamped_piano_roll = unclamp_midi(piano_roll)
        encoded = midi_encode_v2(unclamped_piano_roll, program=program)
        pm_song.instruments.append(encoded.instruments[0])

    return pm_song


if __name__ == "__main__":
    # pass
    instrument_to_idx = limit_instruments()
    idx_to_instrument = {v: k for k, v in instrument_to_idx.items()}
    data = load_all(styles, SEQ_LEN, instrument_to_idx)
    # model, _, _ = train_model(LATENT_DIM, EPOCHS, data)
    model_name = 'changelog_22'
    model = tf.keras.models.load_model('out/models/' + model_name)
    # plot_latent_space(model, instrument_to_idx)
    model.summary()
    model.decoder_block.summary()
    interpolated_seqs = interpolate(model, instrument_to_idx, 1, 2, 4)
    for i, interpolated_seq in enumerate(interpolated_seqs):
        generated_seq = tf.expand_dims(interpolated_seq, axis=0)
        print(np.max(generated_seq), generated_seq.shape)
        selected_instruments = separate_instruments(generated_seq, idx_to_instrument, "GREEDY", threshold=1e-3)
        pm_song = pm.PrettyMIDI()
        for program, piano_roll in selected_instruments:
            unclamped_piano_roll = unclamp_midi(piano_roll)
            encoded = midi_encode_v2(unclamped_piano_roll, program=program)
            pm_song.instruments.append(encoded.instruments[0])

        f = open("out/generated_action_interpolated_1+2_greedy_" + str(i) + ".mid", "w")
        f.close()
        pm_song.write("out/generated_action_interpolated_1+2_greedy_" + str(i) + ".mid")
    # label = np.zeros((1, NUM_STYLES))
    # label[:, 0] = 0.5
    # label[:, 5] = 0.5
    # print(label)
    # pm_song = generate(model, 4, label, idx_to_instrument, "GREEDY")
    # f = open("out/generated_ceva_random_15_greedy.mid", "w")
    # f.close()
    # pm_song.write("out/generated_ceva_random_15_greedy.mid")
    # print(idx_to_instrument)

