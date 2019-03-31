'''
import simulation as sim
import random

true_success_rates = [random.uniform(0.01, 0.04) for _ in range(0, 10)]
trials_per_period = len(true_success_rates) * 100

results = sim.compare_methods(periods=28,
                              true_success_rates=true_success_rates,
                              deviation=0.5,
                              trials_per_period=trials_per_period,
                              max_p=0.1)

results = sim.compare_params(method='split',
                             param='true_success_rates',
                             values=[[0.01, 0.015], [0.01, 0.02]],
                             periods=28,
                             true_success_rates='param',
                             deviation=0,
                             trials_per_period=2000,
                             max_p=0.1,
                             rounding=False)

sim.plot(results['periods'], results['parameters'],
         compare='method', relative=True)
'''

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import split as spl
import bandit as ban


def add_split_results(num_options, trials_per_period, max_p, success_rates,
                      split, period, rounding=True):
    if split.p_value > max_p / (period + 1):
        for j in range(num_options):
            if rounding:
                split.add_results(
                    j, round(trials_per_period / num_options),
                    round(trials_per_period / num_options *
                          success_rates[j]))
            else:
                split.add_results(
                    j, trials_per_period / num_options,
                    trials_per_period / num_options * success_rates[j])
    else:
        best_option = np.where(split.successes == max(split.successes))[0][0]
        if rounding:
            split.add_results(best_option, trials_per_period,
                              round(trials_per_period * max(success_rates)))
        else:
            split.add_results(best_option, trials_per_period,
                              trials_per_period * max(success_rates))
    split.calculate_p_value()
    return split.successes.sum()


def add_bandit_results(num_options, trials_per_period, success_rates,
                       bandit, period, rounding=True):
    if period == 0:
        for j in range(num_options):
            if rounding:
                bandit.add_results(
                    j, round(trials_per_period / num_options),
                    round(trials_per_period / num_options * success_rates[j]))
            else:
                bandit.add_results(
                    j, trials_per_period / num_options,
                    trials_per_period / num_options * success_rates[j])
    else:
        option_shares = bandit.repeat_choice(repetitions=100) / 100
        for j in range(num_options):
            if rounding:
                bandit.add_results(
                    j, round(trials_per_period * option_shares[j]),
                    round(trials_per_period * option_shares[j] *
                          success_rates[j]))
            else:
                bandit.add_results(
                    j, trials_per_period * option_shares[j],
                    trials_per_period * option_shares[j] * success_rates[j])
    return bandit.successes.sum()


def simulate(method, periods, true_success_rates, deviation,
             trials_per_period, max_p=None, rounding=True):

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
    for period in range(periods):
        # Calculate success rates under uncertainty (with deviation)
        success_rates = [np.random.RandomState(period).
                         normal(loc=rate, scale=rate * deviation)
                         for rate in true_success_rates]

        # Add results to Split or Bandit
        if method == 'split':
            successes.append(add_split_results(
                num_options, trials_per_period, max_p, success_rates,
                chooser, period, rounding))
        elif method == 'bandit':
            successes.append(add_bandit_results(
                num_options, trials_per_period, success_rates,
                chooser, period, rounding))

        # Add results to max and base successes
        if period == 0:
            if rounding:
                max_successes = [round(trials_per_period * max(success_rates))]
                base_successes = [np.sum(
                    [round(trials_per_period / num_options * success_rates[i])
                     for i in range(num_options)])]
            else:
                max_successes = [trials_per_period * max(success_rates)]
                base_successes = [np.sum(
                    [trials_per_period / num_options * success_rates[i]
                     for i in range(num_options)])]
        else:
            if rounding:
                max_successes.append(
                    max_successes[-1] +
                    round(trials_per_period * max(success_rates)))
                base_successes.append(base_successes[-1] + np.sum([
                    round(trials_per_period / num_options * success_rates[i])
                    for i in range(num_options)]))
            else:
                max_successes.append(max_successes[-1] +
                                     trials_per_period * max(success_rates))
                base_successes.append(base_successes[-1] + np.sum([
                    trials_per_period / num_options * success_rates[i]
                    for i in range(num_options)]))

    return [successes, max_successes, base_successes]


def compare_params(method, param, values, periods, true_success_rates='param',
                   deviation='param', trials_per_period='param',
                   max_p='param', rounding=True):
    results = []
    for value in values:
        if param == 'max_p':
            successes, optima = simulate(
                method, periods, true_success_rates, deviation,
                trials_per_period, value, rounding)[0:2]
        elif param == 'trials_per_period':
            successes, optima = simulate(
                method, periods, true_success_rates, deviation,
                value, max_p, rounding)[0:2]
        elif param == 'true_success_rates':
            successes, optima = simulate(
                method, periods, value, deviation, trials_per_period,
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
                'true_success_rates': true_success_rates,
                'deviation': deviation,
                'trials_per_period': trials_per_period,
                'max_p': max_p
            },
            'periods': period_values
        }


def compare_methods(periods, true_success_rates, deviation,
                    trials_per_period, max_p):

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
            'periods':
            {
                'max_successes': max_successes,
                'split_successes': split_successes,
                'bandit_successes': bandit_successes,
                'base_successes': base_successes
            }
        }


def plot(periods, parameters, compare='methods', relative=False):
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
    if compare == 'methods':
        plt.title('success_rates: ' +
                  str(round(min(parameters['true_success_rates']), 4)) + '..' +
                  str(round(max(parameters['true_success_rates']), 4)) +
                  ', deviation: ' + str(parameters['deviation']) +
                  ', trials: ' + str(parameters['trials_per_period']) +
                  ', max_p: ' + str(parameters['max_p']),
                  fontsize=10)
    elif compare == 'params':
        plt.title('succes_rates: ' + str(parameters['true_success_rates']) +
                  ', deviation: ' + str(parameters['deviation']) +
                  ', trials: ' + str(parameters['trials_per_period']) +
                  ', max_p: ' + str(parameters['max_p']),
                  fontsize=10)
    plt.xlabel('Periods')
    plt.legend()
    plt.show()
