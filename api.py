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
    picks = pick(data, 'ad')
    return jsonify(picks)


@app.route('/adsets', methods=['POST'])
def adsets():
    data = request.json
    picks = pick(data, 'adset')
    return jsonify(picks)


def pick(payload, dimension, formatted=False):
    data = pd.DataFrame(payload)
    if not formatted:
        data = choose.facebook(data, dimension, 'Impressions', 'Link Clicks')
    [options, data] = choose.process(data)
    bandit = choose.BetaBandit(options=options)
    bandit.bulk_add_results(data)
    return bandit.choices(options=options, repetitions=100, onoff=True)


if __name__ == '__main__':
    app.run()
