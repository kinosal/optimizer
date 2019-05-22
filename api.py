from io import StringIO
import math
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from scipy.stats import beta
import matplotlib.pyplot as plt
import process as pro
import bandit as ban


app = Flask(__name__)


@app.route('/ping', methods=['GET', 'POST'])
def ping():
    """
    Return string to show the server is alive
    """
    return 'Server is here'


@app.route('/', methods=['GET'])
def root():
    """
    Root page with links to simple and CSV format_results
    """
    return render_template('index.html')


@app.route('/json', methods=['POST'])
def json():
    """
    Process Facebook ads JSON with "Ad ID", "Adset ID", "Campaign ID",
    "Impressions", "Link Clicks" and "Purchases" for multiple periods;
    return options with suggested budget share for next period
    """
    data = pd.DataFrame(request.json)
    data = pro.facebook(data, purchase_factor=10)
    [options, data] = pro.reindex_options(data)
    data = pro.add_days(data)
    bandit = add_daily_results(data, num_options=len(options),
                               memory=True, shape='linear', cutoff=28)
    shares = choose(bandit=bandit, accelerate=True)
    options = format_results(options, shares, onoff=False)
    return options.to_json(orient='records')


@app.route('/form', methods=['GET', 'POST'])
def form():
    """
    Provide form with cumulated trial and success inputs for two options,
    return options with suggested budget share for next period
    """

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
        shares = choose(bandit=bandit, accelerate=False)
        options = format_results(options, shares, onoff=False)
        records = options.to_dict('records')
        columns = options.columns.values
        save_plot(bandit)
        return render_template('form_result.html', records=records,
                               columns=columns, plot='/static/images/plot.png')

    return render_template('form.html')


@app.route('/csv', methods=['GET', 'POST'])
def csv():
    """
    Provide form to paste Facebook ads CSV with "Ad ID", "Adset ID",
    "Campaign ID", "Impressions", "Link Clicks" and "Purchases" for multiple
    periods, return options with share or on/off suggestion for next period
    """

    if request.method == 'POST':
        data = pd.read_csv(StringIO(request.form['ads']))
        data = pro.facebook(data, purchase_factor=10)
        [options, data] = pro.reindex_options(data)
        data = pro.add_days(data)
        bandit = add_daily_results(data, num_options=len(options),
                                   memory=True, shape='linear', cutoff=28)
        shares = choose(bandit=bandit, accelerate=True)
        onoff = request.form['onoff'] == 'true'
        options = format_results(options, shares, onoff=onoff)
        if onoff:
            options.columns = ['Ad ID', 'Ad Set ID', 'Campaign ID', 'Ad Status']
            options.replace(True, 'ACTIVE', inplace=True)
            options.replace(False, 'PAUSED', inplace=True)
        else:
            options.columns = ['Ad ID', 'Ad Set ID', 'Campaign ID', 'Ad Share']
            options['Ad Share'] = options['Ad Share'].round(2)
        return options.to_csv(index=False, header=True,
                              line_terminator='<br>', sep='\t')

    return render_template('csv.html')


@app.after_request
def after_request(response):
    """
    Clear server cache after every request to display newest plot image
    """
    response.headers['Cache-Control'] = \
        'no-cache, no-store, must-revalidate, public, max-age=0'
    response.headers['Expires'] = 0
    response.headers['Pragma'] = 'no-cache'
    return response


def add_daily_results(data, num_options, memory, shape, cutoff):
    """
    For each day, add a period with its option results to the Bandit
    """
    bandit = ban.Bandit(num_options, memory, shape, cutoff)
    for _, group in data.groupby('date'):
        bandit.add_period()
        for i in range(len(group)):
            bandit.add_results(option_id=group.iloc[i]['option_id'],
                               trials=group.iloc[i]['trials'],
                               successes=group.iloc[i]['successes'])
    return bandit


def choose(bandit, accelerate):
    """
    Choose best options at current state,
    return each option's suggested share for the next period
    """
    if accelerate:
        choices = int(np.sqrt(bandit.num_options))
        repetitions = math.ceil(bandit.num_options / choices)
    else:
        choices = 1
        repetitions = 100
    shares = bandit.repeat_choice(choices, repetitions) \
        / (choices * repetitions)
    return shares


def format_results(options, shares, onoff):
    """
    Return True/False instead of numeric share for options if desired
    """
    if onoff:
        status = (shares > 0)
        options['status'] = status.tolist()
    else:
        options['share'] = shares.tolist()
    return options


def save_plot(bandit):
    """
    Save plot with bandit options' PDFs (beta distributions)
    """
    x = np.linspace(0, 1, 100)
    for i in range(len(bandit.trials)):
        plt.plot(x, beta.pdf(
            x, bandit.successes[i], bandit.trials[i] - bandit.successes[i]),
                 label='option ' + str(i+1))
    plt.xlabel('Success rate')
    plt.ylabel('Probablity density')
    plt.grid()
    plt.yticks([])
    plt.legend()
    plt.savefig('static/images/plot.png')
    plt.clf()


if __name__ == '__main__':
    app.run()
