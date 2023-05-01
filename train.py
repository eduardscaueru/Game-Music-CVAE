from model import *
from dataset import *
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
    bce = keras.losses.binary_crossentropy(original_seqs, decoded_seqs)
    # kl loss
    kl = kl_loss(z_mu, z_rho)

    return bce, kl


def train(latent_dim, epochs, dataset):
    note_data = dataset[0][0]
    note_target = dataset[0][1]
    style_data = dataset[0][3]
    num_seqs = note_data.shape[0]
    print(note_data.shape)

    cvae = CVAE(latent_dim)
    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    kl_loss_tracker = keras.metrics.Mean(name='kl_loss')
    bce_loss_tracker = keras.metrics.Mean(name='bce_loss')

    label_list = None
    z_mu_list = None

    for epoch in range(epochs):
        label_list = None
        z_mu_list = None

        for start_batch in np.arange(0, num_seqs - num_seqs % BATCH_SIZE, BATCH_SIZE):
            with tf.GradientTape() as tape:
                seqs = note_data[start_batch:start_batch + BATCH_SIZE, :, :, 0]
                style_labels = style_data[start_batch:start_batch + BATCH_SIZE, 1, :]

                # forward pass
                z_mu, z_rho, decoded_seqs = cvae(seqs, style_labels)
                if epoch == EPOCHS - 1:
                    print(decoded_seqs[0, 0, :])

                # compute loss
                bce, kl = elbo(z_mu, z_rho, decoded_seqs, seqs)
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

        # reset metric states
        kl_loss_tracker.reset_state()
        bce_loss_tracker.reset_state()

    return cvae, z_mu_list, label_list


if __name__ == "__main__":
    data = load_all(styles, BATCH_SIZE, SEQ_LEN)
    train(LATENT_DIM, EPOCHS, data)
