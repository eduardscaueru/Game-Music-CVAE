from model import *
from dataset import idx_to_instrument, midi_encode_v2
from generate import generate
import pretty_midi as pm
from dataset import load_all
from keras import backend as K
import matplotlib.pyplot as plt
tf.get_logger().setLevel('ERROR')


def f1_m(y_true, y_pred):
    def recall_m(y_true, y_pred):
        TP = tf.cast(K.sum(K.round(K.clip(y_true * y_pred, 0, 1))), dtype=tf.float32)
        Positives = tf.cast(K.sum(K.round(K.clip(y_true, 0, 1))), dtype=tf.float32)

        recall = TP / (Positives + K.epsilon())
        return recall

    def precision_m(y_true, y_pred):
        TP = tf.cast(K.sum(K.round(K.clip(y_true * y_pred, 0, 1))), dtype=tf.float32)
        Pred_Positives = tf.cast(K.sum(K.round(K.clip(y_pred, 0, 1))), dtype=tf.float32)

        precision = TP / (Pred_Positives + K.epsilon())
        return precision

    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)

    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


# closed form kl loss computation between variational posterior q(z|x) and unit Gaussian prior p(z)
def kl_loss(z_mu, z_rho):
    sigma_squared = tf.math.softplus(z_rho) ** 2
    kl_1d = -0.5 * (1 + tf.math.log(sigma_squared) - z_mu ** 2 - sigma_squared)

    # sum over sample dim, average over batch dim
    kl_batch = tf.reduce_mean(tf.reduce_sum(kl_1d, axis=1))

    return kl_batch


def elbo(z_mu, z_rho, decoded_seqs, original_seqs):
    # reconstruction loss
    # original = tf.cast(original_seqs, decoded_seqs.dtype)
    # original = tf.reshape(original, [-1])
    # decoded = tf.reshape(decoded_seqs, [-1])
    # # bce = keras.losses.binary_crossentropy(original, decoded)
    # # original = original[40:50]
    # # print(original)
    # # original = tf.add(original, tf.constant([0, 1, 0, 1, 0], dtype=tf.float32))
    # bce = tf.nn.sigmoid_cross_entropy_with_logits(logits=decoded, labels=original)
    # # print("Before mean: ", bce)
    # bce = tf.reduce_mean(bce, axis=0)
    # # print(decoded[0, :5], original)
    # # print("After mean: ", bce)
    bce_loss = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM)
    bce = bce_loss(tf.cast(original_seqs, decoded_seqs.dtype), decoded_seqs)
    # print("Bce: ", bce)
    # kl loss
    kl = kl_loss(z_mu, z_rho)
    # print("KL: ", kl)

    return bce, kl


def save_generated_song(model, epoch, length):
    label = np.zeros((1, NUM_STYLES))
    label[:, 0] = 1
    pm_song = generate(model, length, label)
    f = open("out/generated_test_epoch_" + str(epoch) + ".mid", "w")
    f.close()
    pm_song.write("out/generated_test_epoch_" + str(epoch) + ".mid")


def train_model(latent_dim, epochs, dataset):
    note_data = dataset[0][0]
    note_target = dataset[0][1]
    style_data = dataset[0][3]
    num_seqs = note_data.shape[0]
    print("note shape", note_data.shape)
    print("style shape", style_data.shape)
    # print("note_data between", len(note_data[0 < note_data < 1]))

    cvae = CVAE(latent_dim)
    lr = 1.0
    optimizer = keras.optimizers.Adadelta(learning_rate=lr)

    kl_loss_tracker = keras.metrics.Mean(name='kl_loss')
    bce_loss_tracker = keras.metrics.Mean(name='bce_loss')
    f1_tracker = keras.metrics.Mean(name='f1')

    bce_metric = []
    f1_metric = []
    kl_metric = []

    label_list = None
    z_mu_list = None

    for epoch in range(epochs):
        label_list = None
        z_mu_list = None

        for start_batch in np.arange(0, num_seqs - num_seqs % BATCH_SIZE, BATCH_SIZE):
            with tf.GradientTape() as tape:
                seqs = note_data[start_batch:start_batch + BATCH_SIZE, :, :]
                # target_seqs = note_target[start_batch:start_batch + BATCH_SIZE, :, :]
                style_labels = style_data[start_batch:start_batch + BATCH_SIZE, 1, :]

                # forward pass
                z_mu, z_rho, decoded_seqs = cvae(seqs, style_labels)
                if epoch == EPOCHS - 1:
                    print(decoded_seqs[0, 0, :])

                # compute loss
                bce, kl = elbo(z_mu, z_rho, decoded_seqs, seqs)
                # print("sub 0 bce values", len(bce[bce < 0]))
                loss = bce + BETA * kl
                # loss = bce
                # compute F1 score
                f1_score = f1_m(seqs, decoded_seqs)

            gradients = tape.gradient(loss, cvae.variables)
            # print(np.mean([tf.reduce_mean(g) for g in gradients]))

            optimizer.apply_gradients(zip(gradients, cvae.variables))

            kl_loss_tracker.update_state(BETA * kl)
            bce_loss_tracker.update_state(bce)
            f1_tracker.update_state(f1_score)

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
        epoch_kl, epoch_bce, epoch_f1 = kl_loss_tracker.result(), bce_loss_tracker.result(), f1_tracker.result()
        print(f'epoch: {epoch}, bce: {epoch_bce:.4f}, kl_div: {epoch_kl:.4f}, f1: {epoch_f1:.4f}')
        bce_metric.append(epoch_bce)
        f1_metric.append(epoch_f1)
        kl_metric.append(epoch_kl)

        if epoch > 0 and epoch % GENERATE_EVERY_EPOCH == 0:
            save_generated_song(cvae, epoch, 8)
            # cvae.save('out/models/' + "test" + "_epoch_" + str(epoch))

        # reset metric states
        kl_loss_tracker.reset_state()
        bce_loss_tracker.reset_state()
        f1_tracker.reset_state()

    return cvae, z_mu_list, label_list, bce_metric, f1_metric, kl_metric


if __name__ == "__main__":
    print(tf.version.VERSION)
    print(np.version.version)
    data = load_all(styles, BATCH_SIZE, SEQ_LEN)
    cvae, _, _, bce_metric, f1_metric, kl_metric = train_model(LATENT_DIM, EPOCHS, data)
    model_name = "changelog_2"
    cvae.save('out/models/' + model_name)

    plt.plot(
        np.linspace(1, EPOCHS, EPOCHS),
        bce_metric, linewidth=1.0, color="blue"
    )
    plt.plot(
        np.linspace(1, EPOCHS, EPOCHS),
        kl_metric, linewidth=1.0, color="red"
    )
    plt.ylim(top=5)
    plt.title('Model train loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['bce', 'kl'], loc='upper right')
    plt.show()

    plt.plot(
        np.linspace(1, EPOCHS, EPOCHS),
        f1_metric, linewidth=1.0, color="orange"
    )
    plt.title('Model train F1')
    plt.xlabel('epoch')
    plt.ylabel('F1')
    plt.show()
