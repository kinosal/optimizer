"""
Pipeline:
import choosing
data = pd.read_csv('ads.csv')
data = choosing.import_facebook(data=data, purchase_factor=10)
[options, data] = choosing.process(data)
bandit = choosing.BetaBandit(options)
bandit.bulk_add_results(data, cutoff=7, shape='linear')
bandit.choices(options=options, repetitions=100, onoff=False)
# bandit.show_pdfs()
"""

from datetime import date
import numpy as np
import pandas as pd
from scipy.stats import beta
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt


def import_facebook(data, purchase_factor):
    """
    Process dataframe from Facebook CSV with Ad-, Ad Set-, Campaign ID,
    Reporting Ends, Impressions, Link Clicks, Purchases;
    return normalized columns for further processing
    """
    # Set empty and NaN Impressions, Link Clicks and Purchases to 0
    # for calculations
    data = data.replace('', np.nan)
    data['Impressions'].fillna(value=0, downcast='infer', inplace=True)
    data['Link Clicks'].fillna(value=0, downcast='infer', inplace=True)
    data['Purchases'].fillna(value=0, downcast='infer', inplace=True)
    # Create successes column as Link Clicks + Purchases * purchase factor,
    # maximise successes (alpha) to trials = impressions (alpha + beta) for PDF
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
    """
    Create Beta distributions for multiple variables (options) with trials and
    successes (results), provide methods to add results, draw samples from
    distributions and return choices about future trials for each option
    """

    def __init__(self, options, prior=(1.0, 1.0)):
        self.options = options
        self.num_options = len(options)
        self.prior = prior
        self.trials = np.zeros(shape=(self.num_options,), dtype=float)
        self.successes = np.zeros(shape=(self.num_options,), dtype=float)

    def add_results(self, option_id, trials, successes, distance,
                    cutoff=7, shape='linear'):
        """
        Add past results to one option, discounting by function of distance
        defined by cutoff (no relevance after this point) and alternative
        shapes weighing smaller distances
        """
        if cutoff <= 0:
            raise ValueError('cutoff must be greater than 0')
        # Calculate trial and success weights (aggregates)
        # to add to existing values
        if shape == 'linear':
            # Weight (inverse discount) linearly decreases with distance
            trial_weight = trials * max(1 - distance / cutoff, 0)
            success_weight = successes * max(1 - distance / cutoff, 0)
        elif shape == 'degressive':
            # Weight decrease (slope) shrinks with distance
            trial_weight = trials * \
                max(2 * cutoff / (distance + cutoff) - 1, 0)
            success_weight = successes * \
                max(2 * cutoff / (distance + cutoff) - 1, 0)
        elif shape == 'progressive':
            # Weight decrease (slope) grows over distance
            trial_weight = trials * max(-(distance / cutoff) ** 2 + 1, 0)
            success_weight = successes * max(-(distance / cutoff) ** 2 + 1, 0)
        else:
            raise ValueError('shape not recognized')
        # Add weights to existing result values
        self.trials[option_id] = self.trials[option_id] + trial_weight
        self.successes[option_id] = self.successes[option_id] + success_weight

    def bulk_add_results(self, data, cutoff=7, shape='linear'):
        """
        Add multiple results for multiple options from dataframe
        """
        for i in range(len(data)):
            self.add_results(option_id=data.iloc[i]['option_id'],
                             trials=data.iloc[i]['trials'],
                             successes=data.iloc[i]['successes'],
                             distance=data.iloc[i]['days_ago'],
                             cutoff=cutoff, shape=shape)

    def choose_option(self):
        """
        Create Beta distributions for each option, draw one sample each,
        compare and choose the option with the largest value
        """
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
        """
        Repeat option choosing process by Beta distribution sampling and
        return aggregate outcome
        """
        # Initialize option_counts array with zeros
        option_counts = np.zeros(shape=(self.num_options,), dtype=int)
        # Choose best option and increment it's count value
        for _ in range(repetitions):
            option = self.choose_option()
            option_counts[option] += 1
        # Return True/False instead of count numbers if desired
        if onoff:
            option_counts = (option_counts > 0)
        return option_counts

    def choices(self, options, repetitions, onoff):
        """
        Add choices to and return final options dataframe
        """
        choices = self.repeat_choice(repetitions, onoff).tolist()
        options['choice'] = choices
        return options

    # def show_pdfs(self):
    #     x = np.linspace(0, 1, 100)
    #     for i in range(len(self.trials)):
    #         plt.plot(x, beta.pdf(
    #             x, self.successes[i], self.trials[i] - self.successes[i]),
    #                  label='ad' + str(i))
    #     plt.legend()
    #     plt.show()
