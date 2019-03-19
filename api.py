from io import StringIO
from flask import Flask, request
import pandas as pd
import choosing


app = Flask(__name__)


@app.route('/ping', methods=['GET', 'POST'])
def ping():
    return 'Server is here'


@app.route('/csv/ads', methods=['GET', 'POST'])
def ads_form():
    if request.method == 'POST':
        repetitions = int(request.form['budget'])
        data = pd.read_csv(StringIO(request.form['ads']))
        choices = choose(data=data, repetitions=repetitions)
        choices.columns = ['Ad ID', 'Ad Set ID', 'Campaign ID', 'Ad Status']
        choices.replace(True, 'ACTIVE', inplace=True)
        choices.replace(False, 'PAUSED', inplace=True)
        return choices.to_csv(index=False, header=True,
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
    if 'budget' in request.args:
        repetitions = min(int(request.args['budget']), 100)
    else:
        repetitions = 100
    data = pd.DataFrame(request.json)
    choices = choose(data=data, repetitions=repetitions)
    return choices.to_json(orient='records')


def choose(data, purchase_factor=10, repetitions=100,
           formatted=False, onoff=True):
    if not formatted:
        data = choosing.import_facebook(data, purchase_factor)
    [options, data] = choosing.process(data)
    bandit = choosing.BetaBandit(options)
    bandit.bulk_add_results(data)
    return bandit.choices(options, repetitions, onoff)


if __name__ == '__main__':
    app.run()
