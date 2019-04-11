import numpy as np
from scipy.stats import beta


class Bandit():
    def __init__(self, num_options, memory=False,
                 shape='constant', cutoff=28):
        self.memory = memory
        self.num_options = num_options
        self.prior = (1.0, 1.0)
        self.trials = np.zeros(shape=(self.num_options,), dtype=int)
        self.successes = np.zeros(shape=(self.num_options,), dtype=int)
        if self.memory:
            self.periods = {'trials': [], 'successes': []}
            self.shape = shape
            self.cutoff = cutoff

    def add_results(self, option_id, trials, successes):
        self.trials[option_id] = self.trials[option_id] + trials
        self.successes[option_id] = self.successes[option_id] + successes
        if self.memory:
            if option_id == 0:
                self.periods['trials'].append(
                    np.zeros(shape=(self.num_options,), dtype=int))
                self.periods['successes'].append(
                    np.zeros(shape=(self.num_options,), dtype=int))
            self.periods['trials'][-1][option_id] = trials
            self.periods['successes'][-1][option_id] = successes

    def weigh_options(self):
        trial_weights = np.zeros(shape=(self.num_options,), dtype=float)
        success_weights = np.zeros(shape=(self.num_options,), dtype=float)
        num_periods = len(self.periods['trials'])
        for period_id in range(num_periods):
            distance = num_periods - period_id
            for option_id in range(self.num_options):
                trials = self.periods['trials'][period_id][option_id]
                successes = self.periods['successes'][period_id][option_id]
                if distance < self.cutoff:
                    if self.shape == 'constant':
                        weight = 1
                    elif self.shape == 'linear':
                        weight = 1 - distance / self.cutoff
                    elif self.shape == 'degressive':
                        weight = 2 * self.cutoff / (distance + self.cutoff) - 1
                    elif self.shape == 'progressive':
                        weight = -(distance / self.cutoff) ** 2 + 1
                    trial_weights[option_id] += trials * weight
                    success_weights[option_id] += successes * weight
        return [trial_weights, success_weights]

    def choose_options(self, choices):
        if self.memory:
            trial_weights, success_weights = self.weigh_options()
        else:
            trial_weights, success_weights = self.trials, self.successes
        sampled_theta = []
        for i in range(self.num_options):
            # Construct beta distribution for each option's success
            dist = beta(self.prior[0] + success_weights[i],
                        self.prior[1] + trial_weights[i] - success_weights[i])
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
