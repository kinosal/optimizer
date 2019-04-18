from io import StringIO
from flask import Flask, request
import numpy as np
import pandas as pd
import process as pro
import bandit as ban


app = Flask(__name__)


@app.route('/ping', methods=['GET', 'POST'])
def ping():
    return 'Server is here'


@app.route('/json', methods=['POST'])
def json():
    data = pd.DataFrame(request.json)
    data = pro.facebook(data, purchase_factor=10)
    [options, data] = pro.reindex_options(data)
    data = pro.add_days(data)
    bandit = add_daily_results(data, num_options=len(options),
                               memory=True, shape='linear', cutoff=28)
    shares = choose(bandit=bandit)
    options = format_results(options, shares, onoff=False)
    return options.to_json(orient='records')


@app.route('/form', methods=['GET', 'POST'])
def form():
    if request.method == 'POST':
        entries = [value for value in list(request.form.values()) if value]
        num_options = int(len(entries) / 2)
        options = pd.DataFrame(
            [{'option': str(i+1)} for i in range(num_options)])
        trials = [int(entries[i*2]) for i in range(num_options)]
        successes = [int(entries[i*2+1]) for i in range(num_options)]
        bandit = ban.Bandit(num_options=num_options, memory=False)
        for i in range(num_options):
            bandit.add_results(option_id=i, trials=trials[i],
                               successes=successes[i])
        shares = choose(bandit=bandit)
        options = format_results(options, shares, onoff=False)
        return options.to_json(orient='records')

    return '''<form method="POST">
                  <input name="trials_1" type="number" min="1" step="1" placeholder="trials_1" />
                  <input name="successes_1" type="number" min="1" step="1" placeholder="successes_1" />
                  <br>
                  <input name="trials_2" type="number" min="1" step="1" placeholder="trials_2" />
                  <input name="successes_2" type="number" min="1" step="1" placeholder="successes_2" />
                  <br>
                  <input type="submit" value="Submit"><br>
              </form>'''


@app.route('/csv', methods=['GET', 'POST'])
def csv():
    if request.method == 'POST':
        data = pd.read_csv(StringIO(request.form['ads']))
        data = pro.facebook(data, purchase_factor=10)
        [options, data] = pro.reindex_options(data)
        data = pro.add_days(data)
        bandit = add_daily_results(data, num_options=len(options),
                                   memory=True, shape='linear', cutoff=28)
        shares = choose(bandit=bandit)
        options = format_results(options, shares, onoff=True)
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


def add_daily_results(data, num_options, memory, shape, cutoff):
    bandit = ban.Bandit(num_options, memory, shape, cutoff)
    for _, group in data.groupby('date'):
        bandit.add_period()
        for i in range(len(group)):
            bandit.add_results(option_id=group.iloc[i]['option_id'],
                               trials=group.iloc[i]['trials'],
                               successes=group.iloc[i]['successes'])
    return bandit


def choose(bandit):
    choices = int(np.sqrt(bandit.num_options))
    repetitions = int(bandit.num_options / choices) + 2
    shares = bandit.repeat_choice(choices, repetitions) \
        / (choices * repetitions)
    return shares


def format_results(options, shares, onoff):
    if onoff:
        status = (shares > 0)
        options['status'] = status.tolist()
    else:
        options['share'] = shares.tolist()
    return options


if __name__ == '__main__':
    app.run()
