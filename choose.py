"""
[options, data] = process_data()
bb = BetaBandit(num_options=len(options))
bb.bulk_add_results(data)
bb.repeat_choice(100)
"""

import numpy as np
import pandas as pd
from scipy.stats import beta
from datetime import date
import matplotlib.pyplot as plt


def process_data(file='ads.csv'):
    data = pd.read_csv(file)
    # Extract relevant columns
    if file == 'ads.csv':
        data = data[['Ad Name', 'Ad Set Name', 'Campaign Name',
                     'Reporting Ends', 'Impressions', 'Link Clicks']]
    elif file == 'adsets.csv':
        data = data[['Ad Set Name', 'Campaign Name', 'Reporting Ends',
                     'Impressions', 'Link Clicks']]
    # Create and save the ad id for every result
    data['option_id'] = 0
    if file == 'ads.csv':
        options = data[['Ad Name', 'Ad Set Name', 'Campaign Name']]. \
            drop_duplicates().reset_index()
        for i in range(len(data)):
            data.at[i, 'option_id'] = options.loc[
                (options['Ad Name'] == data.iloc[i]['Ad Name']) &
                (options['Ad Set Name'] == data.iloc[i]['Ad Set Name']) &
                (options['Campaign Name'] == data.iloc[i]['Campaign Name'])].index[0]
    elif file == 'adsets.csv':
        options = data[['Ad Set Name', 'Campaign Name']]. \
            drop_duplicates().reset_index()
        for i in range(len(data)):
            data.at[i, 'option_id'] = options.loc[
                (options['Ad Set Name'] == data.iloc[i]['Ad Set Name']) &
                (options['Campaign Name'] == data.iloc[i]['Campaign Name'])].index[0]
    # Calculate and save days ago for every result
    data['Reporting Ends'] = \
        pd.to_datetime(data['Reporting Ends'], format='%Y-%m-%d').dt.date
    data['days_ago'] = 0
    for i in range(len(data)):
        data.at[i, 'days_ago'] = \
            (date.today() - data.iloc[i]['Reporting Ends']).days
    # Set NaN Impressions and Link Clicks to 0
    data['Impressions'].fillna(value=0, inplace=True)
    data['Link Clicks'].fillna(value=0, inplace=True)
    return [options, data]


class BetaBandit():
    def __init__(self, num_options=2, prior=(1.0, 1.0)):
        self.trials = np.zeros(shape=(num_options,), dtype=float)
        self.successes = np.zeros(shape=(num_options,), dtype=float)
        self.num_options = num_options
        self.prior = prior

    def add_results(self, trial_id, trials, successes, discount):
        self.trials[trial_id] = self.trials[trial_id] + \
            trials * (1 - min((discount - 1) / 100, 1))
        self.successes[trial_id] = self.successes[trial_id] + \
            successes * (1 - min((discount - 1) / 100, 1))

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

    def show_pdfs(self):
        x = np.linspace(0, 0.1, 100)
        for i in range(len(self.trials)):
            plt.plot(x, beta.pdf(
                x, self.successes[i], self.trials[i] - self.successes[i]),
                     label='ad' + str(i))
        plt.legend()
        plt.show()
