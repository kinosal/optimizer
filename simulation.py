'''
import simulation as sim
import random

true_rates = [random.uniform(0.01, 0.04) for _ in range(0, 50)]
trials = len(true_rates) * 100

results = sim.compare_methods(
    periods=28, true_rates=true_rates,
    deviation=0.5, change=0, trials=trials, max_p=0.1)

results = sim.compare_params(
    method='bandit', param='deviation',
    values=[0, 0.25, 0.5, 0.75],
    periods=28, true_rates=[0.01, 0.015], deviation='param',
    change=0, trials=2000, max_p=0.1, rounding=True)

sim.plot(results['periods'], results['parameters'], relative=True)
'''

import random
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import split as spl
import bandit as ban


def add_split_results(trials, max_p, rates, split, period, rounding=True):
    if split.p_value > max_p / (period + 1):
        active_options = np.where(split.trials == max(split.trials))[0]
        for option_id in active_options:
            if rounding:
                split.add_results(
                    option_id, round(trials / len(active_options)),
                    round(trials / len(active_options) *
                          rates[option_id]))
            else:
                split.add_results(
                    option_id, trials / len(active_options),
                    trials / len(active_options) *
                    rates[option_id])
    else:
        best_options = np.where(split.successes == max(split.successes))[0]
        for option_id in best_options:
            if rounding:
                split.add_results(
                    option_id, round(trials / len(best_options)),
                    round(trials / len(best_options) *
                          rates[option_id]))
            else:
                split.add_results(
                    option_id, trials / len(best_options),
                    trials / len(best_options) *
                    rates[option_id])
    split.calculate_p_value()
    return split.successes.sum()


def add_bandit_results(num_options, trials, rates, bandit, period,
                       rounding=True):
    if period == 0:
        for j in range(num_options):
            if rounding:
                bandit.add_results(
                    j, round(trials / num_options),
                    round(trials / num_options * rates[j]))
            else:
                bandit.add_results(
                    j, trials / num_options,
                    trials / num_options * rates[j])
    else:
        option_shares = bandit.repeat_choice(repetitions=100) / 100
        for j in range(num_options):
            if rounding:
                bandit.add_results(
                    j, round(trials * option_shares[j]),
                    round(trials * option_shares[j] *
                          rates[j]))
            else:
                bandit.add_results(
                    j, trials * option_shares[j],
                    trials * option_shares[j] * rates[j])
    return bandit.successes.sum()


def simulate(method, periods, true_rates, deviation, change,
             trials, max_p=None, rounding=True):

    num_options = len(true_rates)

    rate_changes = [random.uniform(1 - change, 1 + change)
                    for rate in true_rates]

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
    for period in range(periods):
        # Calculate success rates under uncertainty (with deviation)
        rates = [max(np.random.RandomState((i+1)*(period+1)).normal(
            loc=rate * rate_changes[i] ** period,
            scale=rate * rate_changes[i] ** period * deviation), 0)
                 for i, rate in enumerate(true_rates)]

        # Add results to Split or Bandit
        if method == 'split':
            successes.append(add_split_results(
                trials, max_p, rates,
                chooser, period, rounding))
        elif method == 'bandit':
            successes.append(add_bandit_results(
                num_options, trials, rates,
                chooser, period, rounding))

        # Add results to max and base successes
        if period == 0:
            if rounding:
                max_successes = [round(trials * max(rates))]
                base_successes = [np.sum(
                    [round(trials / num_options * rates[i])
                     for i in range(num_options)])]
            else:
                max_successes = [trials * max(rates)]
                base_successes = [np.sum(
                    [trials / num_options * rates[i]
                     for i in range(num_options)])]
        else:
            if rounding:
                max_successes.append(
                    max_successes[-1] +
                    round(trials * max(rates)))
                base_successes.append(base_successes[-1] + np.sum([
                    round(trials / num_options * rates[i])
                    for i in range(num_options)]))
            else:
                max_successes.append(max_successes[-1] +
                                     trials * max(rates))
                base_successes.append(base_successes[-1] + np.sum([
                    trials / num_options * rates[i]
                    for i in range(num_options)]))

    return [successes, max_successes, base_successes]


def compare_params(method, param, values, periods, true_rates, deviation,
                   change, trials, max_p, rounding=True):
    results = []
    for value in values:
        if param == 'max_p':
            successes, optima = simulate(
                method, periods, true_rates, deviation, change,
                trials, value, rounding)[0:2]
        elif param == 'trials':
            successes, optima = simulate(
                method, periods, true_rates, deviation, change,
                value, max_p, rounding)[0:2]
        elif param == 'true_rates':
            successes, optima = simulate(
                method, periods, value, deviation, change, trials,
                max_p, rounding)[0:2]
        elif param == 'deviation':
            successes, optima = simulate(
                method, periods, true_rates, value, change, trials,
                max_p, rounding)[0:2]
        # Devide successes by optima for relative comparison independent from
        # param values (absolute values will not be returned)
        results.append([suc / opt for suc, opt in zip(successes, optima)])

    period_values = {}
    for i in range(len(values)):
        period_values[str(values[i])] = results[i]
    period_values['max_successes'] = [1.0] * periods

    return \
        {
            'parameters':
            {
                'method': method,
                'param': param,
                'values': values,
                'true_rates': true_rates,
                'deviation': deviation,
                'change': change,
                'trials': trials,
                'max_p': max_p
            },
            'periods': period_values
        }


def compare_methods(periods, true_rates, deviation, change,
                    trials, max_p):

    split_successes, max_successes, base_successes = simulate(
        'split', periods, true_rates, deviation, change, trials, max_p)

    bandit_successes = simulate('bandit', periods, true_rates,
                                deviation, change, trials)[0]

    return \
        {
            'parameters':
            {
                'true_rates': true_rates,
                'deviation': deviation,
                'change': change,
                'trials': trials,
                'max_p': max_p
            },
            'periods':
            {
                'max_successes': max_successes,
                'split_successes': split_successes,
                'bandit_successes': bandit_successes,
                'base_successes': base_successes
            }
        }


def plot(periods, parameters, relative=False):
    x = np.linspace(1, len(periods['max_successes']),
                    len(periods['max_successes']))
    if relative:
        for graph in list(periods):
            plt.plot(x, [x / y for x, y in zip(
                periods[graph], periods['max_successes'])], label=graph)
        plt.ylabel('Cumulatated successes / optimum')
    else:
        for graph in list(periods):
            plt.plot(x, periods[graph], label=graph)
        plt.ylabel('Cumulatated total number of successes')
    if isinstance(parameters['true_rates'], list) and len(
            parameters['true_rates']) > 4:
        plt.title('num_options: ' + str(len(parameters['true_rates'])) +
                  ', true_rates: ' +
                  str(round(min(parameters['true_rates']), 4)) + '..' +
                  str(round(max(parameters['true_rates']), 4)) +
                  ', deviation: ' + str(parameters['deviation']) +
                  ', change: ' + str(parameters['change']) +
                  ', trials: ' + str(parameters['trials']) +
                  ', max_p: ' + str(parameters['max_p']),
                  fontsize=10)
    else:
        plt.title('true_rates: ' + str(parameters['true_rates']) +
                  ', deviation: ' + str(parameters['deviation']) +
                  ', change: ' + str(parameters['change']) +
                  ', trials: ' + str(parameters['trials']) +
                  ', max_p: ' + str(parameters['max_p']),
                  fontsize=10)
    plt.xlabel('Periods')
    plt.legend()
    plt.show()
