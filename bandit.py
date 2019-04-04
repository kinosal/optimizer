import numpy as np
from scipy.stats import beta


class Bandit():
    def __init__(self, num_options):
        self.num_options = num_options
        self.prior = (1.0, 1.0)
        self.trials = np.zeros(shape=(self.num_options,), dtype=float)
        self.successes = np.zeros(shape=(self.num_options,), dtype=float)

    def add_results(self, option_id, trials, successes):
        self.trials[option_id] = self.trials[option_id] + trials
        self.successes[option_id] = self.successes[option_id] + successes

    def choose_options(self, choices):
        sampled_theta = []
        for i in range(self.num_options):
            # Construct beta distribution for each option's success
            dist = beta(self.prior[0] + self.successes[i],
                        self.prior[1] + self.trials[i] - self.successes[i])
            # Draw one sample from beta distribution
            sampled_theta += [dist.rvs()]
        # Return the indices of the samples with the largest values
        return [idx for (idx, theta) in sorted(
            enumerate(sampled_theta), key=lambda x: x[1])[-choices:]]

    def repeat_choice(self, choices=1, repetitions=100):
        # Initialize option_counts array with zeros
        option_counts = np.zeros(shape=(self.num_options,), dtype=int)
        # Choose best option and increment it's count value
        for _ in range(repetitions):
            options = self.choose_options(choices=choices)
            for option in options:
                option_counts[option] += 1
        return option_counts
