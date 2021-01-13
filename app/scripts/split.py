import numpy as np
from scipy.stats import chi2_contingency


class Split:
    """
    Split instance holds cumulated trials, successes and failures
    as well as current p-value; expects num_options to initialize
    """

    def __init__(self, num_options):
        self.num_options = num_options
        self.trials = np.zeros(shape=(self.num_options,), dtype=int)
        self.successes = np.zeros(shape=(self.num_options,), dtype=int)
        self.failures = np.zeros(shape=(self.num_options,), dtype=int)
        self.p_value = 1.0

    def add_results(self, option_id, trials, successes):
        """
        Add trials, successes and failures from last period to one option
        """
        self.trials[option_id] = self.trials[option_id] + trials
        self.successes[option_id] = self.successes[option_id] + successes
        self.failures[option_id] = self.failures[option_id] + trials - successes

    def calculate_p_value(self):
        """
        Calculate p-value from contingency table (observations)
        with successes and failures
        """
        observations = []
        for i in range(self.num_options):
            observations.append([self.successes[i], self.failures[i]])
        self.p_value = chi2_contingency(observed=observations, correction=False)[1]
