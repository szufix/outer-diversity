from src.max_diversity.main import compute_optimal_nodes, compute_ic_threshold
from src.max_diversity.plot import plot_optimal_nodes_results, plot_holy_ic
from time import time
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import re


from src.diversity.diversity_utils import normalization
from src.print_utils import COLOR, LABEL

from src.max_diversity.plot import load_domain_size_data, plot_holy_ic_and_optimal_nodes_results


def compute():

    methods = [
        'smpl_holy_ic',
    ]

    num_candidates = 8

    max_iterations = None
    num_samples = 1000

    for run in range(10):
        print(f'Run {run}')
        for threshold in reversed(range(10,25+1)):
            print(f'Threshold: {threshold}')
            for method_name in methods:
                compute_ic_threshold(num_candidates, method_name, threshold=threshold,
                                      num_samples=num_samples, run=run)





if __name__ == "__main__":

    num_candidates = 8

    base = [
        # 'euclidean_3d',
        'euclidean_2d',
        'spoc',
        'sp_double_forked',
        'caterpillar',
        'balanced',
        'largest_condorcet'
        'single_peaked',
        'single_crossing',
        'euclidean_1d',
        'ext_single_vote',
        'single_vote',
    ]


    methods = [
        'smpl_sa',
        'smpl_ic',
    ]

    domain_sizes = range(1,520+1)

    # max_iterations = None
    # num_samples = 1000

    plot_holy_ic_and_optimal_nodes_results(base,
        num_candidates,
        methods,
        domain_sizes,
        with_structured_domains = True)

