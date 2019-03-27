'''
import simulation

results = simulation.compare_methods(periods=28,
                                     true_success_rates=[0.01, 0.015],
                                     deviation=0.0,
                                     trials_per_period=2000,
                                     max_p=0.1)

results = simulation.compare_params(method='split',
                                    param = 'trials_per_period',
                                    values = [1000, 2000, 3000, 4000],
                                    periods=28,
                                    true_success_rates=[0.01, 0.015],
                                    deviation=0.0,
                                    max_p=0.1)

simulation.plot(results['periods'],
                results['parameters'],
                comparison='relative')
'''

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import split as spl
import bandit as ban


def simulate(method, periods, true_success_rates, deviation,
             trials_per_period, max_p=None):

    num_options = len(true_success_rates)

    # Initialize Split or Bandit instances
    if method == 'split':
        chooser = spl.Split(num_options=num_options)
    elif method == 'bandit':
        chooser = ban.Bandit(num_options=num_options)

    # For each period calculate and add successes for methods as well as
    # the optimal (max) and the random choice (base)
    successes = []
    max_successes = []
    base_successes = []
    for i in range(periods):
        # Calculate success rates under uncertainty (with deviation)
        success_rates = [np.random.RandomState(i).
                         normal(loc=rate, scale=rate * deviation)
                         for rate in true_success_rates]

        # Add results based on and (Bonferroni adjusted) p-value for split
        # and period for bandit
        if (method == 'split' and chooser.p_value > max_p / (i + 1)) \
           or (method == 'bandit' and i == 0):
            for j in range(num_options):
                chooser.add_results(
                    j, round(trials_per_period / num_options),
                    round(trials_per_period / num_options *
                          success_rates[j]))

        elif method == 'split':
            best_option = np.where(chooser.successes ==
                                   max(chooser.successes))[0][0]
            chooser.add_results(
                best_option, trials_per_period,
                round(trials_per_period * max(success_rates)))

        elif method == 'bandit':
            option_percentages = chooser.repeat_choice(repetitions=100) / 100
            for j in range(num_options):
                chooser.add_results(
                    j, round(trials_per_period * option_percentages[j]),
                    round(trials_per_period * option_percentages[j] *
                          success_rates[j]))

        if method == 'split':
            chooser.calculate_p_value()
        successes.append(chooser.successes.sum())

        if i == 0:
            max_successes = [round(trials_per_period * max(success_rates))]
            base_successes = [np.sum(
                [round(trials_per_period / num_options * success_rates[i])
                 for i in range(num_options)])]
        else:
            max_successes.append(max_successes[-1] +
                                 round(trials_per_period * max(success_rates)))
            base_successes.append(base_successes[-1] + np.sum([
                round(trials_per_period / num_options * success_rates[i])
                for i in range(num_options)]))

    return [successes, max_successes, base_successes]


def compare_params(method, param, values, periods, true_success_rates='param',
                   deviation='param', trials_per_period='param',
                   max_p='param'):
    variations = []
    for value in values:
        if param == 'p_value':
            variations.append(simulate(method, periods, true_success_rates,
                                       deviation, trials_per_period, value))
        elif param == 'trials_per_period':
            variations.append(simulate(method, periods, true_success_rates,
                                       deviation, value, max_p))
    periods = {}
    if param == 'trials_per_period':
        for i in range(len(values)):
            periods[str(values[i])] = [x / values[i] for x in variations[i][0]]
        periods['max_successes'] = [x / values[0] for x in variations[0][1]]
    else:
        for i in range(len(values)):
            periods[str(values[i])] = variations[i][0]
        periods['max_successes'] = variations[0][1]
        periods['base_successes'] = variations[0][2]

    return \
        {
            'parameters':
            {
                'method': method,
                'param': param,
                'values': values,
                'true_success_rates': true_success_rates,
                'deviation': deviation,
                'trials_per_period': trials_per_period,
                'max_p': max_p
            },
            'periods': periods
        }


def compare_methods(periods, true_success_rates, deviation,
                    trials_per_period, max_p=None):

    split_successes, max_successes, base_successes = \
        simulate('split', periods, true_success_rates,
                 deviation, trials_per_period, max_p)

    bandit_successes = simulate('bandit', periods, true_success_rates,
                                deviation, trials_per_period)[0]

    return \
        {
            'parameters':
            {
                'true_success_rates': true_success_rates,
                'deviation': deviation,
                'trials_per_period': trials_per_period,
                'max_p': max_p
            },
            'totals':
            {
                'max_successes': max_successes[-1],
                'split_successes': split_successes[-1],
                'bandit_successes': bandit_successes[-1],
                'base_successes': base_successes[-1]
            },
            'periods':
            {
                'max_successes': max_successes,
                'split_successes': split_successes,
                'bandit_successes': bandit_successes,
                'base_successes': base_successes
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
              ', trials_per_period: ' + str(parameters['trials_per_period']) +
              ', max_p: ' + str(parameters['max_p']),
              fontsize=10)
    plt.xlabel('Periods')
    plt.legend()
    plt.show()
