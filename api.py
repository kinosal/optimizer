from io import StringIO
from flask import Flask, request
import numpy as np
import pandas as pd
import process
import bandit as ban


app = Flask(__name__)


@app.route('/ping', methods=['GET', 'POST'])
def ping():
    return 'Server is here'


@app.route('/csv/ads', methods=['GET', 'POST'])
def ads_form():
    if request.method == 'POST':
        data = pd.read_csv(StringIO(request.form['ads']))
        options = choose(data=data, purchase_factor=10,
                         formatted=False, onoff=True)
        options.columns = ['Ad ID', 'Ad Set ID', 'Campaign ID', 'Ad Status']
        options.replace(True, 'ACTIVE', inplace=True)
        options.replace(False, 'PAUSED', inplace=True)
        return options.to_csv(index=False, header=True,
                              line_terminator='<br>', sep='\t')

    return '''<form method="POST">
                  Daily Budget<br>
                  <input name="budget" type="number" min="1" step="1" /><br><br>
                  Ads (CSV)<br>
                  Reporting Starts,Reporting Ends,Ad Name,Ad ID,Campaign Name,Campaign ID,Ad Set Name,Ad Set ID,Impressions,Link Clicks,Purchases<br>
                  <textarea name="ads" rows="50" cols="100"></textarea><br><br>
                  <input type="submit" value="Submit"><br>
              </form>'''


@app.route('/ads', methods=['POST'])
def ads():
    data = pd.DataFrame(request.json)
    options = choose(data=data, purchase_factor=10, formatted=False)
    return options.to_json(orient='records')


def choose(data, purchase_factor=10, formatted=False, onoff=True):
    if not formatted:
        data = process.facebook(data, purchase_factor)
    [options, data] = process.add_option_id(data)
    data = process.add_distance(data)
    bandit = ban.Bandit(num_options=len(options), memory=False,
                        shape='constant', cutoff=28)
    for i in range(len(data)):
        bandit.add_results(option_id=data.iloc[i]['option_id'],
                           trials=data.iloc[i]['trials'],
                           successes=data.iloc[i]['successes'])
    choices = int(np.sqrt(bandit.num_options))
    repetitions = int(bandit.num_options / choices) + 2
    shares = bandit.repeat_choice(choices, repetitions) \
        / (choices * repetitions)
    if onoff:
        status = (shares > 0)
        options['status'] = status.tolist()
    else:
        options['share'] = shares.tolist()
    return options


if __name__ == '__main__':
    app.run()
