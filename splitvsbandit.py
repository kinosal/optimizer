import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
# from scipy.stats import fisher_exact
from scipy.stats import beta


class Split():
    def __init__(self, num_options):
        self.num_options = num_options
        self.trials = np.zeros(shape=(self.num_options,), dtype=float)
        self.successes = np.zeros(shape=(self.num_options,), dtype=float)
        self.failures = np.zeros(shape=(self.num_options,), dtype=float)
        self.p_value = 1.0

    def add_results(self, option_id, trials, successes):
        self.trials[option_id] = self.trials[option_id] + trials
        self.successes[option_id] = self.successes[option_id] + successes
        self.failures[option_id] = \
            self.failures[option_id] + trials - successes

    def calculate_p_value(self):
        observations = []
        for i in range(self.num_options):
            observations.append([self.successes[i], self.failures[i]])
        self.p_value = \
            chi2_contingency(observed=observations, correction=False)[1]
        # Alternative p-value calculation with Fisher's exact test:
        # fisher_exact(table=observations, alternative='two-sided')[1]


class BetaBandit():
    def __init__(self, num_options, prior=(1.0, 1.0)):
        self.num_options = num_options
        self.prior = prior
        self.trials = np.zeros(shape=(self.num_options,), dtype=float)
        self.successes = np.zeros(shape=(self.num_options,), dtype=float)

    def add_results(self, option_id, trials, successes):
        self.trials[option_id] = self.trials[option_id] + trials
        self.successes[option_id] = self.successes[option_id] + successes

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

    def repeat_choice(self, repetitions):
        # Initialize option_counts array with zeros
        option_counts = np.zeros(shape=(self.num_options,), dtype=int)
        # Choose best option and increment it's count value
        for _ in range(repetitions):
            option = self.choose_option()
            option_counts[option] += 1
        return option_counts
