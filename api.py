from io import StringIO
import math
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from scipy.stats import beta
import matplotlib.pyplot as plt
import process as pro
import bandit as ban
import config
import ast
from facebook_business.api import FacebookAdsApi
from facebook_business.adobjects.ad import Ad


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
    Root page with links to simple and CSV form
    """
    return render_template('index.html')


@app.route('/json', methods=['POST'])
def json():
    """
    Process ads JSON with daily breakdown of ad_id, impressions, clicks and
    conversions; return options with suggested status for next period
    """
    data = pd.DataFrame(request.json)
    data = pro.facebook(data, click_weight=1, conversion_weight=10)
    [options, data] = pro.reindex_options(data)
    data = pro.add_days(data)
    bandit = add_daily_results(data, num_options=len(options),
                               memory=True, shape='linear', cutoff=28)
    shares = choose(bandit=bandit, accelerate=True)
    status = request.args.get('status') == 'true'
    options = format_results(options, shares, status=status)
    update = request.args.get('update') == 'true'
    if update:
        app_id = config.APP_ID
        app_secret = config.APP_SECRET
        access_token = config.ACCESS_TOKEN
        updated = update_status(app_id, app_secret, access_token, options)
        return updated.to_json(orient='records')
    return options.to_json(orient='records')


@app.route('/form', methods=['GET', 'POST'])
def form():
    """
    Provide form with cumulated trial and success inputs for multiple options,
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
        options = format_results(options, shares, status=False)
        records = options.to_dict('records')
        columns = options.columns.values
        save_plot(bandit)
        return render_template('form_result.html', records=records,
                               columns=columns, plot='/static/images/plot.png')

    return render_template('form.html')


@app.route('/csv', methods=['GET', 'POST'])
def csv():
    """
    Provide form to paste ads CSV with daily breakdown of ad_id, impressions,
    clicks and conversions; return options with suggested budget share or
    status for next period
    """

    if request.method == 'POST':
        if request.form['update'] == 'true':
            app_id = request.form['app_id']
            app_secret = request.form['app_secret']
            access_token = request.form['access_token']
            channels = ast.literal_eval(request.form['channels'])
            records = ast.literal_eval(request.form['records'])
            updatable = ['facebook', 'instagram']
            indices = []
            for channel in updatable:
                if channel in channels:
                    indices.append(channels.index(channel))
            results = pd.DataFrame(columns=['ad_id', 'ad_status'])
            for index in indices:
                for record in records[index]:
                    results.loc[len(results)] = \
                        [record['ad_id'], record['ad_status']]
            updated = update_status(app_id, app_secret, access_token, results)
            return updated.to_csv(index=False, header=True,
                                  line_terminator='<br>', sep='\t')

        data = pd.read_csv(StringIO(request.form['ads']), sep=None)
        data = pro.preprocess(data, int(request.form['click_weight']),
                              int(request.form['conversion_weight']))
        [options, data] = pro.reindex_options(data)
        data = pro.add_days(data)
        bandit = add_daily_results(data, num_options=len(options),
                                   memory=True, shape='linear', cutoff=28)
        shares = choose(bandit=bandit, accelerate=True)

        output = request.form['output']
        if output == 'status':
            results = format_results(options, shares, status=True)
        elif output == 'share':
            results = format_results(options, shares, status=False).round(2)

        if 'channel' in options.columns:
            channel_shares = format_results(options, shares, status=False). \
                groupby('channel')['ad_share'].sum().round(2)
            channels = []
            records = []
            for name, group in results.groupby('channel'):
                channels.append(name)
                group = group.drop(['channel'], axis=1)
                columns = group.columns.values
                records.append(group.to_dict('records'))
            return render_template('csv_result_channels.html',
                                   channels=channels,
                                   channel_shares=channel_shares,
                                   records=records, columns=columns)

        records = results.to_dict('records')
        columns = results.columns.values
        return render_template('csv_result.html', records=records,
                               columns=columns)

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


def format_results(options, shares, status):
    """
    Return ACTIVE/PAUSED instead of numeric share for options if desired
    """
    results = options.copy()
    if status:
        status = (shares > 0)
        results['ad_status'] = status.tolist()
        results.replace(True, 'ACTIVE', inplace=True)
        results.replace(False, 'PAUSED', inplace=True)
    else:
        results['ad_share'] = shares.tolist()
        results['ad_share'] = results['ad_share']
    return results


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


def update_status(app_id, app_secret, access_token, options):
    """
    Update status of ads on Facebook if different from respective suggestion;
    return dataframe with updated ads
    """
    api = FacebookAdsApi.init(app_id, app_secret, access_token)
    updated = []
    # Determine number of required batches since
    # Facebook only allows 50 API calls per batch
    num_options = len(options.index)
    batch_size = 50
    batches = int(num_options / batch_size) + 1
    # Split options into batches and loop through those
    i = 0
    for _ in range(batches):
        option_batch = options.loc[i:i+batch_size, :]
        i += batch_size
        update_batch = api.new_batch()
        # Loop through options within batch, compare current and suggested
        # ad status and update if changed
        for _, row in option_batch.iterrows():
            ad_id = str(row['ad_id'])
            ad = Ad(ad_id)
            ad.api_get(fields=[Ad.Field.status])
            old_status = ad[Ad.Field.status]
            new_status = row['ad_status']
            if old_status != new_status:
                ad[Ad.Field.status] = new_status
                updated.append([ad_id, old_status + ' -> ' + new_status])
                ad.api_update(batch=update_batch, fields=[],
                              params={Ad.Field.status: new_status})
        update_batch.execute()
    return pd.DataFrame(updated, columns=['ad_id', 'updated'])


if __name__ == '__main__':
    app.run()
