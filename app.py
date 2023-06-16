from flask import Flask, render_template, Response, request
import os
from midi_util import limit_instruments
from dataset import load_all
from constants import styles
from train import train_model


app = Flask(__name__)
num_requests = 0
saved_models = []
config = {}
hidden_layers = []
data = None


@app.route("/dataset", methods=['GET', 'POST'])
def dataset():
    global hidden_layers
    global data
    global config
    global saved_models

    print("Dataset")
    form = request.form.to_dict()
    config.update(form)
    print(config)
    instrument_to_idx = None
    if "ips" in config and config["ips"] != '':
        instrument_to_idx = limit_instruments(max_instruments_per_song=int(config["ips"]))
    else:
        instrument_to_idx = limit_instruments()

    if "get_num_instruments" in form:
        config["num_instruments"] = len(instrument_to_idx) - 1
    else:
        notes_per_bar = int(config["notes_per_beat"]) * int(config["beats_per_bar"]) * int(config["note_ts"])
        seq_len = int(float(config["bars"]) * notes_per_bar)
        config["seq_len"] = seq_len
        min_note = int(config["min_note"])
        max_note = int(config["max_note"])
        instruments_per_song = int(config["ips"])
        fs = int(config["freq"])
        num_instruments = int(config["num_instruments"])
        num_notes_instrument = max_note - min_note + 1
        config["num_notes"] = (num_instruments + 1) * num_notes_instrument

        data = load_all(styles, seq_len, instrument_to_idx, min_note=min_note, max_note=max_note,
                        instruments_per_song=instruments_per_song, fs=fs, num_instruments=num_instruments)
        print(data[0][0].shape)
        config["data_shape"] = str(data[0][0].shape)

    return render_template('index.html', saved_models=saved_models, config=config, hidden_layers=hidden_layers)


@app.route("/train", methods=['GET', 'POST'])
def train():
    global hidden_layers
    global saved_models
    global config

    print("Train")
    form = request.form.to_dict()
    config.update(form)
    print(config)

    hidden_layers = []
    to_delete = []
    for k, v in config.items():
        if "layer_" in k:
            hidden_layers.append(int(v))
            to_delete.append(k)
    for x in to_delete:
        config.pop(x)
    print(hidden_layers)

    seq_len = int(config["seq_len"])
    num_notes = int(config["num_notes"])
    latent_dim = int(config["latent_dim"])
    batch_size = int(config["batch_size"])
    epochs = int(config["epochs"])
    name = config["name"]
    generate_every_epoch = int(config["generate_every_epoch"])
    beta = config["beta"]
    if config["beta"] != "sigmoid" and config["beta"] != "cosine" and config["beta"] != "linear":
        beta = float(config["beta"])

    learning_rate = float(config["learning_rate"])
    optimizer = config["optimizer"]

    cvae, _, _, bce_metric, f1_metric, kl_metric = train_model(data, hidden_layers, latent_dim=latent_dim, epochs=epochs,
                                                               batch_size=batch_size, name=name,
                                                               generate_every_epoch=generate_every_epoch,
                                                               beta_strategy=beta, learning_rate=learning_rate,
                                                               keras_optimizer=optimizer,
                                                               seq_len=seq_len, num_notes=num_notes)

    return render_template('index.html', saved_models=saved_models, config=config, hidden_layers=hidden_layers)


@app.route("/generate", methods=['GET', 'POST'])
def generate():
    global hidden_layers
    global saved_models
    global config
    print("Generate")
    form = request.form.to_dict()
    config.update(form)
    print(config)

    return render_template('index.html', saved_models=saved_models, config=config, hidden_layers=hidden_layers)


@app.route("/", methods=['GET', 'POST'])
def index():
    global hidden_layers
    global num_requests
    global config
    global saved_models
    num_requests += 1
    if num_requests == 1:
        i = 0
        for file in os.listdir("out/models/"):
            saved_models.append({"value": i, "name": file})

        return render_template('index.html', saved_models=saved_models, config=config, hidden_layers=hidden_layers)

    form = request.form.to_dict()
    config.update(form)
    print(config)

    return render_template('index.html', saved_models=saved_models, config=config, hidden_layers=hidden_layers)


if __name__ == '__main__':
    app.run()

