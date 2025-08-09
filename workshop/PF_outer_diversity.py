import csv

import matplotlib.pyplot as plt
import numpy as np

from src.clustering import vote2pote,swap_distance_between_potes
from src.domain import *
from src.election import Election
from src.print_utils import NICE_NAME

samplers = {
    'euclidean_3d': euclidean_3d_domain,
    'euclidean_2d': euclidean_2d_domain,
    'euclidean_1d': euclidean_1d_domain,
    'caterpillar': group_separable_caterpillar_domain,
    'balanced': group_separable_balanced_domain,
    'single_peaked': single_peaked_domain,
    'single_crossing': single_crossing_domain,
    'sp_double_forked': sp_double_forked_domain,
    'spoc': spoc_domain,
}




def compute_diversity(num_candidates, n_iter):
    results = []
    for sampler_name, sampler in samplers.items():
#        print(f'Processing {sampler_name} with {num_candidates} candidates.')
        output_mean = []
        output_std = []
        distances = []
        votes = sampler(num_candidates=num_candidates)
        votes = [np.array(v) for v in votes]
        domain = [ vote2pote(vote, num_candidates) for vote in votes]
        for _ in range(n_iter):
            ic_vote = np.random.permutation(num_candidates)
            ic_pote = vote2pote( ic_vote, num_candidates )
            dst = min( [swap_distance_between_potes( ic_pote, pote, num_candidates) for pote in domain])
#            print(dst)
            distances.append( dst )
        mean = np.mean(np.array(distances))
        std = np.std(np.array(distances))
        results.append((mean,std,sampler_name))
        print(f'{sampler_name} with {num_candidates} candidates: {mean} +/- {std}.')
    print()
    print()
    print(results)
    results.sort()
    for i in range(len(results)):
        print( f"{results[i][2]}: {results[i][0]} +/- {results[i][1]}")



            #    if sampler_name in CONDORCET_DOMAIN:
            #        search_space = e.votes
            #    else:
            #        ic = np.array([np.random.permutation(num_candidates) for _ in range(ic_size)])
            #        search_space = np.concatenate((e.votes, ic), axis=0)







num_candidates = 8
n_iter = 10000

compute_diversity(num_candidates, n_iter)

base = [
    'impartial',
    'caterpillar',
    'euclidean_3d',
    'euclidean_2d',
    'spoc',
    'balanced',
    'sp_double_forked',
    'single_crossing',
    'euclidean_1d',
]


