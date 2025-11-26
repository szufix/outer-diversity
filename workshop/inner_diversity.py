import csv

import matplotlib.pyplot as plt
import numpy as np

from src.clustering import local_search_kKemeny_single_k_from_domain
from src.domain import *
from src.election import Election
from src.print_utils import *

samplers = {
    # 'euclidean_3d': euclidean_3d_domain,
    # 'euclidean_2d': euclidean_2d_domain,
    # 'euclidean_1d': euclidean_1d_domain,
    # 'caterpillar': group_separable_caterpillar_domain,
    # 'balanced': group_separable_balanced_domain,
    # 'single_peaked': single_peaked_domain,
    # 'single_crossing': single_crossing_domain,
    # 'sp_double_forked': sp_double_forked_domain,
    # 'spoc': spoc_domain,
    # 'largest_condorcet': largest_condorcet_domain,
    'ball': ball_domain
}

CONDORCET_DOMAIN = [
    'euclidean_1d',
    'caterpillar',
    'balanced',
    'single_peaked',
    'single_crossing',
    'sp_double_forked',
    'largest_condorcet',
]



YLIM = {
    6: [-0.5,8],  # 7.5
    8: [-0.5,14.5], # 14
    10: [-0.5,23],# 22.5
    12: [-0.5,33.5],# 33
}

def export_data_to_csv(sampler_name, dict_mean, dict_std, num_candidates):
    with open(f'data/diversity/domain/{sampler_name}_stats_m{num_candidates}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["key", "mean", 'std'])  # header
        mean = [float(value) for value in dict_mean]
        std =  [float(value) for value in dict_std]
        writer.writerow((sampler_name, mean, std))

def import_data_from_csv(sampler_name, num_candidates):
    mean = {}
    std = {}

    with open(f'data/diversity/domain/{sampler_name}_stats_m{num_candidates}.csv', mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            mean = [float(x) for x in row["mean"].strip('[]').split(',')]
            std = [float(x) for x in row["std"].strip('[]').split(',')]

    return mean, std


def compute_diversity(num_candidates, max_k, n_iter, ic_size):

    for sampler_name, sampler in samplers.items():
        print(f'Processing {sampler_name} with {num_candidates} candidates.')
        output_mean = []
        output_std = []
        for k in range(1, max_k + 1):
            print(f'k = {k}')
            scores = []
            for _ in range(n_iter):

                e = Election(0, num_candidates)
                e.votes = sampler(num_candidates=num_candidates)
                num_voters = len(e.votes)
                e.num_voters = num_voters

                if sampler_name in CONDORCET_DOMAIN:
                    search_space = e.votes
                else:
                    ic = np.array([np.random.permutation(num_candidates) for _ in range(ic_size)])
                    search_space = np.concatenate((e.votes, ic), axis=0)

                score = local_search_kKemeny_single_k_from_domain(e, search_space, k, 1)

                scores.append(score['value'] / num_voters)

            output_mean.append(np.mean(np.array(scores)))
            output_std.append(np.std(np.array(scores)))

        export_data_to_csv(sampler_name, output_mean, output_std, num_candidates)


def inner_diversity(domain, num_candidates, max_k=5, n_iter=1, ic_size=512):


    for k in range(1, max_k + 1):
        print(f'k = {k}')
        scores = []
        for _ in range(n_iter):

            e = Election(0, num_candidates)
            e.votes = domain
            num_voters = len(e.votes)
            e.num_voters = num_voters

            ic = np.array([np.random.permutation(num_candidates) for _ in range(ic_size)])
            search_space = np.concatenate((e.votes, ic), axis=0)

            score = local_search_kKemeny_single_k_from_domain(e, search_space, k, 1)

            scores.append(score['value'] / num_voters)


