'''
import simulation
results = simulation.experiment(periods=28,
                                true_success_rates=[0.01, 0.015, 0.02, 0.025],
                                deviation=0.1,
                                trials_per_period=2000,
                                max_p=0.05)
simulation.plot(results['periods'],
                results['parameters'],
                comparison='relative')
'''

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import splitvsbandit


def experiment(periods, true_success_rates=[0.01, 0.015], deviation=0.0,
               trials_per_period=1000, max_p=0.05):
    num_options = len(true_success_rates)

    # Initialize Split and Bandit instances
    split = splitvsbandit.Split(num_options=num_options)
    bandit = splitvsbandit.BetaBandit(num_options=num_options)

    # Calculate success rates under uncertainty (with deviation)
    # for first period
    success_rates = [np.random.normal(loc=rate, scale=rate * deviation)
                     for rate in true_success_rates]

    # Add results for first period
    for i in range(num_options):
        split.add_results(i, round(trials_per_period / num_options),
                          round(trials_per_period / num_options *
                                success_rates[i]))
        bandit.add_results(i, round(trials_per_period / num_options),
                           round(trials_per_period / num_options *
                                 success_rates[i]))

    split.calculate_p_value()

    max_successes = [round(trials_per_period * max(success_rates))]
    base_successes = [np.sum([round(trials_per_period / num_options *
                                    success_rates[i])
                              for i in range(num_options)])]
    split_successes = [split.successes.sum()]
    bandit_successes = [bandit.successes.sum()]

    # For each period calculate and add trials and successes
    for i in range(periods-1):
        # Calculate success rates with deviation for this period
        success_rates = [np.random.normal(loc=rate, scale=rate * deviation)
                         for rate in true_success_rates]

        # Optimal choice
        max_successes.append(
            max_successes[-1] + round(trials_per_period * max(success_rates)))

        # Base (random) choice
        base_successes.append(base_successes[-1] + np.sum([
            round(trials_per_period / num_options * success_rates[i])
            for i in range(num_options)]))

        # Split
        # Adjust max_p with Bonferroni method to allow sequential testing
        # if split.p_value > max_p / (i + 2):
        # Alternatively use Sidak correction
        # if split.p_value > 1 - (1 - max_p) ** (1 / (i + 2)):
        if split.p_value > 1 - (1 - max_p) ** (1 / (i + 2)):
            for j in range(num_options):
                split.add_results(j, round(trials_per_period / num_options),
                                  round(trials_per_period / num_options *
                                        success_rates[j]))
        else:
            best_option = np.where(split.successes ==
                                   max(split.successes))[0][0]
            split.add_results(best_option, trials_per_period,
                              round(trials_per_period * max(success_rates)))
        split.calculate_p_value()
        split_successes.append(split.successes.sum())

        # Bandit
        option_percentages = bandit.repeat_choice(repetitions=100) / 100
        for j in range(num_options):
            bandit.add_results(
                j, round(trials_per_period * option_percentages[j]),
                round(trials_per_period * option_percentages[j] *
                      success_rates[j]))
        bandit_successes.append(bandit.successes.sum())

    return \
        {
            'parameters':
            {
                'true_success_rates': true_success_rates,
                'deviation': deviation,
                'trials_per_period': trials_per_period
            },
            'totals':
            {
                'max_successes': max_successes[-1],
                'split_successes': split.successes.sum(),
                'bandit_successes': bandit.successes.sum(),
                'base_successes': base_successes[-1]
            },
            'periods':
            {
                'max_successes': max_successes,
                'split_successes': split_successes,
                'bandit_successes': bandit_successes,
                'base_successes': base_successes
            },
            'choosers':
            {
                'split': split,
                'bandit': bandit
            }
        }


def plot(periods, parameters, comparison='absolute'):
    x = np.linspace(1, len(periods['max_successes']),
                    len(periods['max_successes']))
    if comparison == 'absolute':
        for graph in list(periods):
            plt.plot(x, periods[graph], label=graph)
        plt.ylabel('Absolute total number of successes')
    elif comparison == 'relative':
        for graph in list(periods):
            plt.plot(x, [x / y for x, y in zip(
                periods[graph], periods['max_successes'])], label=graph)
        plt.ylabel('Cumulatated successes as percentage of optimal choice')
    plt.suptitle(
        'Simulation of competing options choosen by different algorithms',
        fontsize=12)
    plt.title('true_success_rates: ' + str(parameters['true_success_rates']) +
              ', deviation: ' + str(parameters['deviation']) +
              ', trials_per_period: ' + str(parameters['trials_per_period']),
              fontsize=10)
    plt.xlabel('Periods')
    plt.legend()
    plt.show()
