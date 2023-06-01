from model import *
from dataset import idx_to_instrument, midi_encode_v2
from generate import generate_song
import pretty_midi as pm
from dataset import load_all
tf.get_logger().setLevel('ERROR')


# closed form kl loss computation between variational posterior q(z|x) and unit Gaussian prior p(z)
def kl_loss(z_mu, z_rho):
    sigma_squared = tf.math.softplus(z_rho) ** 2
    kl_1d = -0.5 * (1 + tf.math.log(sigma_squared) - z_mu ** 2 - sigma_squared)

    # sum over sample dim, average over batch dim
    kl_batch = tf.reduce_mean(tf.reduce_sum(kl_1d, axis=1))

    return kl_batch


def elbo(z_mu, z_rho, decoded_seqs, original_seqs):
    # reconstruction loss
    # x = decoded_seqs[0, 0, :]
    # for i in range(x.shape[0]):
    #     y = x[i]
    #     # print(y)
    #     print(len([v for v in y if 1 < v < 0]))
    # print(decoded_seqs[0, 0, :])
    # print([x for x in sorted(decoded_seq[0, 0, :128], reverse=True)])
    # print(tf.reduce_max(x) - 0.0001)
    # y = tf.where(tf.reduce_max(decoded_seqs) - 0.001 < decoded_seqs, 1., 0.)
    # print(y)
    # print("sub 0 decoded values", len(decoded_seqs[decoded_seqs < 0]))
    bce = keras.losses.binary_crossentropy(tf.cast(original_seqs, decoded_seqs.dtype), decoded_seqs)
    # kl loss
    kl = kl_loss(z_mu, z_rho)

    return bce, kl


def train_model(latent_dim, epochs, dataset):
    note_data = dataset[0][0]
    note_target = dataset[0][1]
    style_data = dataset[0][3]
    num_seqs = note_data.shape[0]
    print("note shape", note_data.shape)
    print("style shape", style_data.shape)
    # print("note_data between", len(note_data[0 < note_data < 1]))

    cvae = CVAE(latent_dim)
    optimizer = keras.optimizers.Adam()

    kl_loss_tracker = keras.metrics.Mean(name='kl_loss')
    bce_loss_tracker = keras.metrics.Mean(name='bce_loss')

    label_list = None
    z_mu_list = None

    for epoch in range(epochs):
        label_list = None
        z_mu_list = None

        for start_batch in np.arange(0, num_seqs - num_seqs % BATCH_SIZE, BATCH_SIZE):
            with tf.GradientTape() as tape:
                seqs = note_data[start_batch:start_batch + BATCH_SIZE, :, :]
                target_seqs = note_target[start_batch:start_batch + BATCH_SIZE, :, :]
                style_labels = style_data[start_batch:start_batch + BATCH_SIZE, 1, :]

                flat_seq = seqs.flatten()
                for f in flat_seq:
                    if 0 < f < 1:
                        print("nu e bine")
                        break

                # forward pass
                z_mu, z_rho, decoded_seqs = cvae(seqs, style_labels)
                if epoch == EPOCHS - 1:
                    print(decoded_seqs[0, 0, :])

                # compute loss
                bce, kl = elbo(z_mu, z_rho, decoded_seqs, seqs)
                # print("sub 0 bce values", len(bce[bce < 0]))
                loss = bce + BETA * kl

                gradients = tape.gradient(loss, cvae.variables)

                optimizer.apply_gradients(zip(gradients, cvae.variables))

                kl_loss_tracker.update_state(BETA * kl)
                bce_loss_tracker.update_state(bce)

                # save encoded means and labels for latent space visualization
                if label_list is None:
                    label_list = style_labels
                else:
                    label_list = np.concatenate((label_list, style_labels))

                if z_mu_list is None:
                    z_mu_list = z_mu
                else:
                    z_mu_list = np.concatenate((z_mu_list, z_mu), axis=0)

        # generate new samples
        # generate_conditioned_digits(model, dataset_mean, dataset_std)

        # display metrics at the end of each epoch.
        epoch_kl, epoch_bce = kl_loss_tracker.result(), bce_loss_tracker.result()
        print(f'epoch: {epoch}, bce: {epoch_bce:.4f}, kl_div: {epoch_kl:.4f}')

        if epoch > 0 and epoch % GENERATE_EVERY_EPOCH == 0:
            label = np.zeros((1, NUM_STYLES))
            label[:, 2] = 0.5
            label[:, 6] = 0.5
            generated = generate_song(cvae, 2, label)
            print(np.max(generated))
            print(len(generated[generated > 0.1]))
            print(generated.shape)

            t = 0
            final = np.zeros(
                (NUM_INSTRUMENTS + 1, generated.shape[0] * generated.shape[1], NUM_NOTES_INSTRUMENT))
            instrument_max_probs = {i: 0 for i in range(NUM_INSTRUMENTS + 1)}
            print(final.shape)
            for bars in range(generated.shape[0]):
                for time_step in range(generated.shape[1]):
                    for i in range(NUM_INSTRUMENTS + 1):
                        instrument_seq = generated[bars, time_step,
                                         i * NUM_NOTES_INSTRUMENT:(i + 1) * NUM_NOTES_INSTRUMENT]
                        selected_note_idx = np.argmax(instrument_seq)
                        max_prob = np.max(instrument_seq)

                        instrument_max_probs[i] += max_prob

                        final[i, t, selected_note_idx] = 1
                    t += 1

            final.tofile('out/generated.dat')

            sorted_instruments = sorted(instrument_max_probs.items(), key=lambda x: x[1], reverse=True)
            print(sorted_instruments)
            selected_instruments = [(idx_to_instrument[x[0]], final[x[0], :, :]) for x in sorted_instruments[:3]]

            # selected_instruments = []
            # for instrument_idx in range(NUM_INSTRUMENTS + 1):
            #     if np.sum(final[instrument_idx, :, :, 1]) > 0:
            #         print(instrument_idx, idx_to_instrument[instrument_idx])
            #         selected_instruments.append((idx_to_instrument[instrument_idx], final[instrument_idx, :, :, :]))

            pm_song = pm.PrettyMIDI()
            for program, piano_roll in selected_instruments:
                encoded = midi_encode_v2(piano_roll, program=program)
                pm_song.instruments.append(encoded.instruments[0])

            f = open("out/generated_test_epoch_" + str(epoch) + ".mid", "w")
            f.close()
            pm_song.write("out/generated_test_epoch_" + str(epoch) + ".mid")

        # reset metric states
        kl_loss_tracker.reset_state()
        bce_loss_tracker.reset_state()

    return cvae, z_mu_list, label_list


if __name__ == "__main__":
    print(tf.version.VERSION)
    print(np.version.version)
    data = load_all(styles, BATCH_SIZE, SEQ_LEN)
    cvae, _, _ = train_model(LATENT_DIM, EPOCHS, data)
    model_name = "test"
    cvae.save('out/models/' + model_name)
