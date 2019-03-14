"""
data = pd.read_json('ads.json')
data = facebook(data, 'ad', 'Impressions', 'Link Clicks')
[options, data] = process(data)
bandit = BetaBandit(options=options)
bandit.bulk_add_results(data)
bandit.repeat_choice(repetitions=100)
"""

from datetime import date
import numpy as np
import pandas as pd
from scipy.stats import beta
# import matplotlib.pyplot as plt


def facebook(data, dimension, trials, successes):
    # Extract and rename relevant columns
    if dimension == 'ad':
        data = data[['Ad ID', 'Reporting Ends', trials, successes]]
    elif dimension == 'adset':
        data = data[['Ad Set ID', 'Reporting Ends', trials, successes]]
    data.columns = ['id', 'date', 'trials', 'successes']
    return data


def process(data):
    """
    Processes dataframe with id, date, trials and successes and
    returns distinct id options as well as dataframe with Bandit results
    """
    # Assign new IDs starting from 0 to options to use as indices for Bandit
    options = data['id'].drop_duplicates().reset_index()
    data['option_id'] = 0
    for i in range(len(data)):
        data.at[i, 'option_id'] = options.loc[
            options['id'] == data.iloc[i]['id']].index[0]
    # Calculate and save days ago for every result
    data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d').dt.date
    data['days_ago'] = 0
    for i in range(len(data)):
        data.at[i, 'days_ago'] = (date.today() - data.iloc[i]['date']).days
    # Drop results that are older than 28 days
    data = data[data['days_ago'] <= 28]
    # Set empty and NaN Impressions and Link Clicks to 0
    data = data.replace('', np.nan)
    data['trials'].fillna(value=0, inplace=True)
    data['successes'].fillna(value=0, inplace=True)
    return [options, data]


class BetaBandit():
    def __init__(self, options, prior=(1.0, 1.0)):
        self.options = options
        self.num_options = len(options)
        self.prior = prior
        self.trials = np.zeros(shape=(self.num_options,), dtype=float)
        self.successes = np.zeros(shape=(self.num_options,), dtype=float)

    def add_results(self, option_id, trials, successes, distance, cutoff=28):
        self.trials[option_id] = self.trials[option_id] + \
            trials * max(1 - distance / cutoff, 0)
        self.successes[option_id] = self.successes[option_id] + \
            successes * max(1 - distance / cutoff, 0)

    def bulk_add_results(self, data):
        for i in range(len(data)):
            self.add_results(data.iloc[i]['option_id'],
                             data.iloc[i]['trials'],
                             data.iloc[i]['successes'],
                             data.iloc[i]['days_ago'])

    def choose_option(self):
        sampled_theta = []
        for i in range(self.num_options):
            # Construct beta distribution for posterior
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
            option_counts = (option_counts > 0).astype(int)
        return option_counts

    def choices(self, options, repetitions, onoff):
        ids = options['id'].apply(str)
        choices = self.repeat_choice(repetitions, onoff).tolist()
        return dict(zip(ids, choices))

    # def show_pdfs(self):
    #     x = np.linspace(0, 0.1, 100)
    #     for i in range(len(self.trials)):
    #         plt.plot(x, beta.pdf(
    #             x, self.successes[i], self.trials[i] - self.successes[i]),
    #                  label='ad' + str(i))
    #     plt.legend()
    #     plt.show()
