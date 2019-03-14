"""
[options, data] = process_data(file='ads.csv', level='ad')
bb = BetaBandit(num_options=max(data.option_id)+1)
bb.bulk_add_results(data)
bb.repeat_choice(repetitions=100)
"""

import numpy as np
import pandas as pd
from scipy.stats import beta
from datetime import date
# import matplotlib.pyplot as plt


def process_data(file, level):
    data = pd.read_csv(file)
    # Extract relevant columns
    if level == 'ad':
        data = data[['Ad ID', 'Reporting Ends', 'Impressions', 'Link Clicks']]
    elif level == 'adset':
        data = data[['Ad Set ID', 'Reporting Ends',
                     'Impressions', 'Link Clicks']]
    # Assign new IDs starting from 0 to ads and ad sets respectively
    # to use as indices for BetaBandit options
    data['option_id'] = 0
    if file == 'ads.csv':
        options = data['Ad ID'].drop_duplicates().reset_index()
        for i in range(len(data)):
            data.at[i, 'option_id'] = options.loc[
                options['Ad ID'] == data.iloc[i]['Ad ID']].index[0]
    elif file == 'adsets.csv':
        options = data['Ad Set ID'].drop_duplicates().reset_index()
        for i in range(len(data)):
            data.at[i, 'option_id'] = options.loc[
                options['Ad Set ID'] == data.iloc[i]['Ad Set ID']].index[0]
    # Calculate and save days ago for every result
    data['Reporting Ends'] = \
        pd.to_datetime(data['Reporting Ends'], format='%Y-%m-%d').dt.date
    data['days_ago'] = 0
    for i in range(len(data)):
        data.at[i, 'days_ago'] = \
            (date.today() - data.iloc[i]['Reporting Ends']).days
    # Drop results that are older than 28 days
    data = data[data['days_ago'] <= 28]
    # Set NaN Impressions and Link Clicks to 0
    data['Impressions'].fillna(value=0, inplace=True)
    data['Link Clicks'].fillna(value=0, inplace=True)
    return [options, data]


class BetaBandit():
    def __init__(self, num_options, prior=(1.0, 1.0)):
        self.trials = np.zeros(shape=(num_options,), dtype=float)
        self.successes = np.zeros(shape=(num_options,), dtype=float)
        self.num_options = num_options
        self.prior = prior

    def add_results(self, option_id, trials, successes, distance, cutoff=28):
        self.trials[option_id] = self.trials[option_id] + \
            trials * max(1 - distance / cutoff, 0)
        self.successes[option_id] = self.successes[option_id] + \
            successes * max(1 - distance / cutoff, 0)

    def bulk_add_results(self, data):
        for i in range(len(data)):
            self.add_results(data.iloc[i]['option_id'],
                             data.iloc[i]['Impressions'],
                             data.iloc[i]['Link Clicks'],
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

    def repeat_choice(self, repetitions):
        option_counts = np.zeros(shape=(self.num_options,), dtype=int)
        for _ in range(repetitions):
            option = self.choose_option()
            option_counts[option] += 1
        return option_counts

    # def show_pdfs(self):
    #     x = np.linspace(0, 0.1, 100)
    #     for i in range(len(self.trials)):
    #         plt.plot(x, beta.pdf(
    #             x, self.successes[i], self.trials[i] - self.successes[i]),
    #                  label='ad' + str(i))
    #     plt.legend()
    #     plt.show()
