from flask import Flask, render_template, Response, request


app = Flask(__name__)
num_requests = 0


@app.route("/", methods=['GET', 'POST'])
def index():
    global num_requests
    num_requests += 1
    if num_requests == 1:
        return render_template('index.html')

    train_form = request.form.to_dict()
    print(request.form)

    return render_template('index.html')
    # return render_template('index.html', freq=train_form['FS'], ips=train_form['INSTRUMENTS_PER_SONG'],
    #                        mgi=train_form['MAX_GENERATED_INSTRUMENTS'], ni=train_form["NUM_INSTRUMENTS"])


if __name__ == '__main__':
    app.run()

