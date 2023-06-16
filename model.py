import tensorflow as tf
from tensorflow import keras
import numpy as np
from constants import *
from keras.layers import LSTM, Dense, RepeatVector, LayerNormalization


def sampling(z_mean, z_std, batch_size, latent_dim):
    # z_mean, z_std = args
    epsilon = keras.backend.random_normal(shape=(batch_size, latent_dim))
    return z_mean + tf.exp(0.5 * z_std) * epsilon


def get_encoder(hidden_layers, batch_size, seq_len, num_notes):
    # Embedding layer pentru a reduce dim NUM_NOTES
    inputs = tf.keras.Input(shape=(seq_len, num_notes), batch_size=batch_size)
    # lstm_output_1 = LSTM(ENCODER_UNITS, return_sequences=True)(inputs)
    # layer_norm_1 = LayerNormalization()(lstm_output_1)
    # lstm_output_2 = LSTM(ENCODER_UNITS_2, return_sequences=True)(layer_norm_1)
    # layer_norm_2 = LayerNormalization()(lstm_output_2)
    # lstm_output_3 = LSTM(ENCODER_UNITS_3)(layer_norm_2)
    # layer_norm_3 = LayerNormalization()(lstm_output_3)
    encoder_output = inputs
    for i, units in enumerate(hidden_layers):
        if i != len(hidden_layers) - 1:
            lstm_output = LSTM(units, return_sequences=True)(encoder_output)
            encoder_output = LayerNormalization()(lstm_output)
        else:
            encoder_output = LSTM(units)(encoder_output)

    return keras.Model(inputs=inputs, outputs=[encoder_output])


def get_latent(latent_dim, last_units, batch_size):
    inputs = tf.keras.Input(shape=(last_units + NUM_STYLES,), batch_size=batch_size)
    mu = Dense(units=latent_dim)(inputs)
    sigma = Dense(units=latent_dim)(inputs)

    return keras.Model(inputs=inputs, outputs=[mu, sigma])


def get_decoder(latent_dim, hidden_layers, batch_size, seq_len, num_notes):
    inputs = tf.keras.Input(shape=(latent_dim + NUM_STYLES,), batch_size=batch_size)
    repeated_inputs = RepeatVector(seq_len)(inputs)
    # lstm_outputs_1 = LSTM(ENCODER_UNITS_3, return_sequences=True)(repeated_inputs)
    # layer_norm_1 = LayerNormalization()(lstm_outputs_1)
    # lstm_outputs_2 = LSTM(ENCODER_UNITS_2, return_sequences=True)(layer_norm_1)
    # layer_norm_2 = LayerNormalization()(lstm_outputs_2)
    # lstm_outputs_3 = LSTM(ENCODER_UNITS, return_sequences=True)(layer_norm_2)
    # layer_norm_3 = LayerNormalization()(lstm_outputs_3)
    # lstm_outputs = LSTM(NUM_NOTES, return_sequences=True, activation="sigmoid")(layer_norm_3) # sterge sigmoid ul
    # Dense layer pentru a creste dim la NUM_NOTES
    reversed_hidden_layers = hidden_layers[::-1]
    decoder_output = repeated_inputs
    for i, units in enumerate(reversed_hidden_layers):
        lstm_output = LSTM(units, return_sequences=True)(decoder_output)
        decoder_output = LayerNormalization()(lstm_output)

    decoder_output = LSTM(num_notes, return_sequences=True, activation="sigmoid")(decoder_output)

    return keras.Model(inputs=inputs, outputs=[decoder_output])


class CVAE(keras.Model):

    def __init__(self, latent_dim, hidden_layers, batch_size=BATCH_SIZE, seq_len=SEQ_LEN, num_notes=NUM_NOTES):
        super().__init__()
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_notes = num_notes
        self.encoder_block = get_encoder(hidden_layers, batch_size, seq_len, num_notes)
        self.latent_block = get_latent(latent_dim, hidden_layers[len(hidden_layers) - 1], batch_size)
        self.decoder_block = get_decoder(latent_dim, hidden_layers, batch_size, seq_len, num_notes)

    def call(self, np_seq, np_labels):
        seq = tf.convert_to_tensor(np_seq, dtype=tf.float32)
        labels = tf.convert_to_tensor(np_labels, dtype=tf.float32)
        # encoder q(z|x,y)
        enc1_output = self.encoder_block(seq)
        # concat feature maps and one hot label vector
        img_lbl_concat = tf.concat([enc1_output, labels], 1)
        z_mu, z_rho = self.latent_block(img_lbl_concat)

        z = sampling(z_mu, z_rho, self.batch_size, self.latent_dim)

        # decoder p(x|z,y)
        z_lbl_concat = tf.concat([z, labels], 1)
        decoded_seq = self.decoder_block(z_lbl_concat)

        return z_mu, z_rho, decoded_seq


if __name__ == "__main__":
    # get_encoder().summary()
    # get_latent(LATENT_DIM).summary()
    # get_decoder(LATENT_DIM).summary()

    model = CVAE(LATENT_DIM, [512, 256])
    model(tf.random.normal((BATCH_SIZE, SEQ_LEN, NUM_NOTES)), tf.zeros((BATCH_SIZE, NUM_STYLES)))
    model.summary()
    print(model.encoder_block.summary())
    print(model.latent_block.summary())
    print(model.decoder_block.summary())
