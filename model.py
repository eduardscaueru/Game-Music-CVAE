import tensorflow as tf
from tensorflow import keras
import numpy as np
from constants import *
from keras.layers import LSTM, Dense, TimeDistributed, InputLayer, Lambda, RepeatVector


def sampling(z_mean, z_std):
    # z_mean, z_std = args
    epsilon = keras.backend.random_normal(shape=(BATCH_SIZE, LATENT_DIM))
    return z_mean + tf.exp(0.5 * z_std) * epsilon


def get_encoder():
    # Embedding layer pentru a reduce dim NUM_NOTES
    inputs = tf.keras.Input(shape=(SEQ_LEN, NUM_NOTES), batch_size=BATCH_SIZE)
    lstm_output = LSTM(ENCODER_UNITS)(inputs)

    return keras.Model(inputs=inputs, outputs=[lstm_output])


def get_latent(latent_dim):
    inputs = tf.keras.Input(shape=(ENCODER_UNITS + NUM_STYLES,), batch_size=BATCH_SIZE)
    mu = Dense(units=latent_dim)(inputs)
    sigma = Dense(units=latent_dim)(inputs)

    return keras.Model(inputs=inputs, outputs=[mu, sigma])


def get_decoder(latent_dim):
    inputs = tf.keras.Input(shape=(latent_dim + NUM_STYLES,), batch_size=BATCH_SIZE)
    repeated_inputs = RepeatVector(SEQ_LEN)(inputs)
    lstm_outputs = LSTM(NUM_NOTES, return_sequences=True, activation="sigmoid")(repeated_inputs)
    # Dense layer pentru a creste dim la NUM_NOTES

    return keras.Model(inputs=inputs, outputs=[lstm_outputs])


class CVAE(keras.Model):

    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder_block = get_encoder()
        self.latent_block = get_latent(latent_dim)
        self.decoder_block = get_decoder(latent_dim)

    def call(self, seq, labels):
        # encoder q(z|x,y)
        enc1_output = self.encoder_block(seq)
        # concat feature maps and one hot label vector
        img_lbl_concat = np.concatenate((enc1_output, labels), axis=1)
        z_mu, z_rho = self.latent_block(img_lbl_concat)

        z = sampling(z_mu, z_rho)

        # decoder p(x|z,y)
        z_lbl_concat = np.concatenate((z, labels), axis=1)
        decoded_seq = self.decoder_block(z_lbl_concat)

        return z_mu, z_rho, decoded_seq


if __name__ == "__main__":
    # get_encoder().summary()
    # get_latent(LATENT_DIM).summary()
    # get_decoder(LATENT_DIM).summary()

    model = CVAE(LATENT_DIM)
    model(tf.random.normal((BATCH_SIZE, SEQ_LEN, NUM_NOTES)), tf.zeros((BATCH_SIZE, NUM_STYLES)))
    model.summary()
    print(model.decoder_block.output.shape[1])
    print(ENCODER_UNITS)
    ENCODER_UNITS = 256
    print(ENCODER_UNITS)
