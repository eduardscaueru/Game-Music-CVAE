from flask import Flask, render_template, Response, request
import os


app = Flask(__name__)
num_requests = 0
saved_models = []
config = {}
hidden_layers = []


@app.route("/dataset", methods=['GET', 'POST'])
def dataset():
    global hidden_layers

    print("Dataset")
    form = request.form.to_dict()
    config.update(form)
    print(config)

    return render_template('index.html', saved_models=saved_models, config=config, hidden_layers=hidden_layers)


@app.route("/train", methods=['GET', 'POST'])
def train():
    global hidden_layers

    print("Train")
    form = request.form.to_dict()
    config.update(form)
    print(config)

    hidden_layers = []
    to_delete = []
    for k, v in config.items():
        if "layer_" in k:
            hidden_layers.append(v)
            to_delete.append(k)
    for x in to_delete:
        config.pop(x)
    print(hidden_layers)

    return render_template('index.html', saved_models=saved_models, config=config, hidden_layers=hidden_layers)


@app.route("/generate", methods=['GET', 'POST'])
def generate():
    global hidden_layers
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
    num_requests += 1
    if num_requests == 1:
        i = 0
        for file in os.listdir("data"):
            saved_models.append({"value": i, "name": file})

        return render_template('index.html', saved_models=saved_models, config=config, hidden_layers=hidden_layers)

    form = request.form.to_dict()
    config.update(form)
    print(config)

    return render_template('index.html', saved_models=saved_models, config=config, hidden_layers=hidden_layers)


if __name__ == '__main__':
    app.run()

