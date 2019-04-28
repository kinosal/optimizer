import numpy as np
from scipy.stats import chi2_contingency
# from scipy.stats import fisher_exact


class Split():
    def __init__(self, num_options):
        self.num_options = num_options
        self.trials = np.zeros(shape=(self.num_options,), dtype=int)
        self.successes = np.zeros(shape=(self.num_options,), dtype=int)
        self.failures = np.zeros(shape=(self.num_options,), dtype=int)
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
