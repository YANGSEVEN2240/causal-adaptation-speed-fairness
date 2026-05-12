import os
import pickle

import numpy as np

from categorical.experiment_loops import (
    experiment_guess, experiment_optimize, experiment_optimize_fairness
)


def optimize_distances(k=10):
    results = []
    base_experiment = {
        'n': 100, 'k': k, 'T': 1500, 'n0': 10,
        'batch_size': 1, 'scheduler_exponent': 0,
        'concentration': 1, 'intervention': 'cause'
    }
    # for lr in [0.01, 0.05, 0.1]:
    for lr in [0.01, 0.1, .5]:
        trajectory = experiment_optimize(
            lr=lr, **base_experiment)
        experiment = {**base_experiment, 'lr': lr, **trajectory}
        results.append(experiment)

    savedir = 'results'
    os.makedirs(savedir, exist_ok=True)
    savefile = os.path.join(savedir, f'categorical_optimize_k={k}.pkl')
    if os.path.exists(savefile):
        with open(savefile, 'rb') as fin:
            previous_results = pickle.load(fin)
    else:
        previous_results = []

    with open(savefile, 'wb') as fout:
        pickle.dump(previous_results + results, fout)


def parameter_sweep(intervention, k, init, seed=17, guess=False, savedir='categorical_results'):
    print(f'intervention on {intervention} with k={k}')
    results = []
    base_experiment = {
        'n': 100, 'k': k, 'T': 1500,
        'batch_size': 1,
        'intervention': intervention,
        'is_init_dense': init,
        'concentration': 1,
        'use_map': True
    }
    for exponent in [0]:
        for lr, n0 in zip([.03, .1, .3, 1, 3, 9, 30],
                          [0.3, 1, 3, 10, 30, 90, 200]):
            np.random.seed(seed)
            parameters = {'n0': n0, 'lr': lr, 'scheduler_exponent': exponent, **base_experiment}
            if guess:
                trajectory = experiment_guess(**parameters)
            else:
                trajectory = experiment_optimize(**parameters)
            results.append({
                'hyperparameters': parameters,
                'trajectory': trajectory,
                'guess': guess
            })

    os.makedirs(savedir, exist_ok=True)

    savefile = f'{intervention}_k={k}.pkl'
    if base_experiment['is_init_dense']:
        savefile = 'denseinit_' + savefile
    else:
        savefile = 'sparseinit_' + savefile
    if guess:
        savefile = 'guess_' + savefile
    else:
        savefile = 'sweep2_' + savefile

    savepath = os.path.join(savedir, savefile)
    with open(savepath, 'wb') as fout:
        pickle.dump(results, fout)


def fairness_parameter_sweep(intervention, k_a, k_x, k_y, init, seed=17, savedir='fairness_results'):
    """Parameter sweep for three-variable fairness models."""
    print(f'Fairness: intervention on {intervention} with k_a={k_a}, k_x={k_x}, k_y={k_y}')
    results = []
    base_experiment = {
        'n': 100, 'k_a': k_a, 'k_x': k_x, 'k_y': k_y, 'T': 1500,
        'batch_size': 1,
        'intervention': intervention,
        'is_init_dense': init,
        'concentration': 1,
    }
    for exponent in [0]:
        for lr in [.03, .1, .3, 1, 3, 9, 30]:
            np.random.seed(seed)
            parameters = {'lr': lr, 'scheduler_exponent': exponent, **base_experiment}
            trajectory = experiment_optimize_fairness(**parameters)
            results.append({
                'hyperparameters': parameters,
                'trajectory': trajectory,
            })

    os.makedirs(savedir, exist_ok=True)

    savefile = f'fairness_{intervention}_ka={k_a}_kx={k_x}_ky={k_y}.pkl'
    if base_experiment['is_init_dense']:
        savefile = 'denseinit_' + savefile
    else:
        savefile = 'sparseinit_' + savefile

    savepath = os.path.join(savedir, savefile)
    with open(savepath, 'wb') as fout:
        pickle.dump(results, fout)


if __name__ == "__main__":
    guess = False
    for init_dense in [True, False]:
        for k in [20]:
            # 原有的双变量实验
            # parameter_sweep('cause', k, init_dense, guess=guess)
            # parameter_sweep('effect', k, init_dense, guess=guess)
            # parameter_sweep('singlecond', k, init_dense)
            # parameter_sweep('gmechanism', k, init_dense)

            # 新增的三变量公平性实验
            # 示例：干预敏感变量 A
            fairness_parameter_sweep('A', k_a=5, k_x=5, k_y=5, init=init_dense)
            # 示例：干预变量 X
            # fairness_parameter_sweep('X', k_a=5, k_x=5, k_y=5, init=init_dense)
            # 示例：同时干预 A 和 X
            # fairness_parameter_sweep('AX', k_a=5, k_x=5, k_y=5, init=init_dense)
            # 示例：干预机制 Y
            # fairness_parameter_sweep('Y', k_a=5, k_x=5, k_y=5, init=init_dense)
