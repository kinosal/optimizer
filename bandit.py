import numpy as np
from scipy.stats import beta


class Bandit():
    """
    Bandit instance holds cumulated trials and successes
    as well as "memorized" periodic results if specified;
    expects num_options and prior tuple to initialize
    """
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

    def add_period(self):
        """
        Add new empty period to memory
        """
        self.periods['trials'].append(
            np.zeros(shape=(self.num_options,), dtype=int))
        self.periods['successes'].append(
            np.zeros(shape=(self.num_options,), dtype=int))

    def add_results(self, option_id, trials, successes):
        """
        Add trials and successes from last period to one option
        """
        self.trials[option_id] += trials
        self.successes[option_id] += successes
        if self.memory:
            self.periods['trials'][-1][option_id] += trials
            self.periods['successes'][-1][option_id] += successes

    def weigh_options(self):
        """
        Weigh options for current period's choice based on distance from now
        with alternatively shaped discount functions
        """
        # Initialize trial and success weights for each option
        trial_weights = np.zeros(shape=(self.num_options,), dtype=float)
        success_weights = np.zeros(shape=(self.num_options,), dtype=float)
        num_periods = len(self.periods['trials'])
        # Loop over each period and option and
        # determine and add respective weights
        for period_id in range(num_periods):
            distance = num_periods - period_id
            for option_id in range(self.num_options):
                trials = self.periods['trials'][period_id][option_id]
                successes = self.periods['successes'][period_id][option_id]
                if distance < self.cutoff:
                    if self.shape == 'constant':
                        # Equal weight for every period
                        weight = 1
                    elif self.shape == 'linear':
                        # Weight linearly decreases with distance
                        weight = 1 - distance / self.cutoff
                    elif self.shape == 'degressive':
                        # Weight decrease (slope) shrinks with distance
                        weight = 2 * self.cutoff / (distance + self.cutoff) - 1
                    elif self.shape == 'progressive':
                        # Weight decrease (slope) grows over distance
                        weight = -(distance / self.cutoff) ** 2 + 1
                    trial_weights[option_id] += trials * weight
                    success_weights[option_id] += successes * weight
        return [trial_weights, success_weights]

    def choose_options(self, choices):
        """
        Create Beta distributions for each option, draw one sample each,
        choose the specified number of options with the largest values
        """
        # Calculate trials and success weighs for options if Bandit has memory,
        # else use cumulated trials and successes
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

    def repeat_choice(self, choices, repetitions):
        """
        Repeat choosing process and return aggregate outcome
        """
        # Initialize option_counts array with zeros
        option_counts = np.zeros(shape=(self.num_options,), dtype=int)
        # Choose best options and increment respective counts
        for _ in range(repetitions):
            options = self.choose_options(choices=choices)
            for option in options:
                option_counts[option] += 1
        return option_counts
