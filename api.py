from flask import Flask, request, jsonify
import pandas as pd
import choose


app = Flask(__name__)


@app.route('/ping', methods=['POST'])
def ping():
    return jsonify({'ping': 'successful'})


@app.route('/ads', methods=['POST'])
def ads():
    data = request.json
    if 'budget' in request.args:
        budget = min(int(request.args['budget']), 100)
    else:
        budget = 100
    picks = pick(data=data, dimension='ad', repetitions=budget)
    return jsonify(picks)


@app.route('/adsets', methods=['POST'])
def adsets():
    data = request.json
    picks = pick(data, 'adset')
    return jsonify(picks)


def pick(data, dimension, purchase_factor=10, repetitions=100,
         formatted=False, onoff=True):
    data = pd.DataFrame(data)
    if not formatted:
        data = choose.facebook(data, dimension, purchase_factor)
    [options, data] = choose.process(data)
    bandit = choose.BetaBandit(options)
    bandit.bulk_add_results(data)
    return bandit.choices(options, repetitions, onoff)


if __name__ == '__main__':
    app.run(debug=True)
