from train import *


def generate_song(cvae, length, style_label):
    generated_seqs = np.zeros((length, cvae.decoder_block.output.shape[1], cvae.decoder_block.output.shape[2]))

    for i in range(length):
        z = tf.random.normal(shape=(1, cvae.latent_block.output[0].shape[1]), mean=0.0, stddev=1.0)
        z_lbl_concat = np.concatenate((z, style_label), axis=1)
        predicted_seq = cvae.decoder_block(z_lbl_concat)

        generated_seqs[i, :, :] = predicted_seq

    return generated_seqs


if __name__ == "__main__":
    data = load_all(styles, BATCH_SIZE, SEQ_LEN)
    model, _, _ = train(LATENT_DIM, EPOCHS, data)
    label = np.zeros((1, NUM_STYLES))
    label[:, 0] = 1
    generated = generate_song(model, 2, label)
    print(np.max(generated))
    print(len(generated[generated > 0.1]))
    print(generate_song(model, 2, label).shape)
