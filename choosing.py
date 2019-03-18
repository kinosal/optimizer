"""
import choosing
data = pd.read_csv('ads.csv')
data = choosing.facebook(data=data, purchase_factor=10)
[options, data] = choosing.process(data)
bandit = choosing.BetaBandit(options)
bandit.bulk_add_results(data)
bandit.choices(options=options, repetitions=10, onoff=True)
"""

from datetime import date
import numpy as np
import pandas as pd
from scipy.stats import beta
# import matplotlib.pyplot as plt


def facebook(data, purchase_factor):
    # Set empty and NaN Impressions, Link Clicks and Purchases to 0
    # for calculations
    data = data.replace('', np.nan)
    data['Impressions'].fillna(value=0, downcast='infer', inplace=True)
    data['Link Clicks'].fillna(value=0, downcast='infer', inplace=True)
    data['Purchases'].fillna(value=0, downcast='infer', inplace=True)
    # Create successes column as Link Clicks + Purchases * purchase factor,
    # maximise successes (alpha) to trials = impressions (beta) for PDF
    data['successes'] = [min(row['Link Clicks'] +
                             row['Purchases'] * purchase_factor,
                             row['Impressions'])
                         for index, row in data.iterrows()]
    # Extract and rename relevant columns
    data = data[['Ad ID', 'Ad Set ID', 'Campaign ID',
                 'Reporting Ends', 'Impressions', 'successes']]
    data.columns = ['ad_id', 'adset_id', 'campaign_id',
                    'date', 'trials', 'successes']
    return data


def process(data, cutoff=7):
    """
    Process dataframe with ad-, adset-, campaign id, date, trials, successes;
    return distinct id options as well as dataframe with Bandit results
    """
    # Assign new IDs starting from 0 to options to use as indices for Bandit
    options = data[['ad_id', 'adset_id', 'campaign_id']] \
        .drop_duplicates().reset_index().drop('index', axis='columns')
    data['option_id'] = 0
    for i in range(len(data)):
        data.at[i, 'option_id'] = options.loc[
            options['ad_id'] == data.iloc[i]['ad_id']].index[0]
    # Calculate and save days ago for every result
    data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d').dt.date
    data['days_ago'] = 0
    for i in range(len(data)):
        data.at[i, 'days_ago'] = (date.today() - data.iloc[i]['date']).days
    # Drop results that are older than the cutoff
    data = data[data['days_ago'] <= cutoff]
    return [options, data]


class BetaBandit():
    def __init__(self, options, prior=(1.0, 1.0)):
        self.options = options
        self.num_options = len(options)
        self.prior = prior
        self.trials = np.zeros(shape=(self.num_options,), dtype=float)
        self.successes = np.zeros(shape=(self.num_options,), dtype=float)

    def add_results(self, option_id, trials, successes, distance, cutoff=7):
        self.trials[option_id] = self.trials[option_id] + \
            trials * max(1 - distance / cutoff / 2, 0)
        self.successes[option_id] = self.successes[option_id] + \
            successes * max(1 - distance / cutoff / 2, 0)

    def bulk_add_results(self, data):
        for i in range(len(data)):
            self.add_results(data.iloc[i]['option_id'],
                             data.iloc[i]['trials'],
                             data.iloc[i]['successes'],
                             data.iloc[i]['days_ago'])

    def choose_option(self):
        sampled_theta = []
        for i in range(self.num_options):
            # Construct beta distribution for each option's success
            dist = beta(self.prior[0] + self.successes[i],
                        self.prior[1] + self.trials[i] - self.successes[i])
            # Draw one sample from beta distribution
            sampled_theta += [dist.rvs()]
        # Return the index of the sample with the largest value
        return sampled_theta.index(max(sampled_theta))

    def repeat_choice(self, repetitions, onoff=False):
        option_counts = np.zeros(shape=(self.num_options,), dtype=int)
        for _ in range(repetitions):
            option = self.choose_option()
            option_counts[option] += 1
        if onoff:
            option_counts = (option_counts > 0)
        return option_counts

    def choices(self, options, repetitions, onoff):
        choices = self.repeat_choice(repetitions, onoff).tolist()
        options['choice'] = choices
        return options

    # def show_pdfs(self):
    #     x = np.linspace(0, 0.1, 100)
    #     for i in range(len(self.trials)):
    #         plt.plot(x, beta.pdf(
    #             x, self.successes[i], self.trials[i] - self.successes[i]),
    #                  label='ad' + str(i))
    #     plt.legend()
    #     plt.show()
