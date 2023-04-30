import tensorflow as tf
from tensorflow import keras
import numpy as np
from constants import *
from keras.layers import LSTM, Dense, TimeDistributed, InputLayer, Lambda, RepeatVector


def sampling(args):
    z_mean, z_std = args
    epsilon = keras.backend.random_normal(shape=(BATCH_SIZE, LATENT_DIM))
    return z_mean + tf.exp(0.5 * z_std) * epsilon


def get_encoder():
    # Embedding layer pentru a reduce dim NUM_NOTES
    inputs = tf.keras.Input(shape=(SEQ_LEN, NUM_NOTES), batch_size=BATCH_SIZE)
    lstm_output = LSTM(ENCODER_UNITS)(inputs)

    return keras.Model(inputs=inputs, outputs=[lstm_output])


def get_latent(latent_dim):
    inputs = tf.keras.Input(shape=(ENCODER_UNITS + 5,), batch_size=BATCH_SIZE)
    mu = Dense(units=latent_dim)(inputs)
    sigma = Dense(units=latent_dim)(inputs)
    z = Lambda(sampling)([mu, sigma])

    return keras.Model(inputs=inputs, outputs=[z])


def get_decoder(latent_dim):
    inputs = tf.keras.Input(shape=(latent_dim + 5,), batch_size=BATCH_SIZE)
    repeated_inputs = RepeatVector(SEQ_LEN)(inputs)
    print(inputs.shape)
    lstm_outputs = LSTM(NUM_NOTES, return_sequences=True)(repeated_inputs)
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
        z = self.latent_block(img_lbl_concat)

        # decoder p(x|z,y)
        z_lbl_concat = np.concatenate((z, labels), axis=1)
        print(z_lbl_concat.shape)
        decoded_seq = self.decoder_block(z_lbl_concat)

        return decoded_seq


if __name__ == "__main__":
    # get_encoder().summary()
    # get_latent(LATENT_DIM).summary()
    # get_decoder(LATENT_DIM).summary()

    model = CVAE(LATENT_DIM)
    model(tf.random.normal((BATCH_SIZE, SEQ_LEN, NUM_NOTES)), tf.zeros((BATCH_SIZE, 5)))
    model.summary()
