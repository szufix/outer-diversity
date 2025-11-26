#!/usr/bin/env python3
import sys
import argparse
import os

from src.domain import ball_domain, single_peaked_domain
from src.diversity.sampling import outer_diversity_sampling

import math
import matplotlib.pyplot as plt

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
    'largest_condorcet': largest_condorcet_domain,
}

import csv

import matplotlib.pyplot as plt
import numpy as np

from src.clustering import local_search_kKemeny_single_k_from_domain
from src.domain import *
from src.election import Election
from src.print_utils import *

def compute_results(num_candidates, max_radius=3, num_samples=100):

    results = []
    for r in range(5, max_radius + 1):
        domain = ball_domain(num_candidates, radius=r)
        outer, _ = outer_diversity_sampling(domain, num_samples=num_samples, inner_distance='swap')
        results.append((r, len(domain), outer))

    return results

def get_sp_div(num_candidates=8, num_samples=100):
    domain = single_peaked_domain(num_candidates)
    outer, sampled = outer_diversity_sampling(domain, num_samples=num_samples, inner_distance='swap')
    return outer


if __name__ == '__main__':
    num_candidates = 8
    sp_diversity = get_sp_div(num_candidates)
    results = compute_results(num_candidates, max_radius=7, num_samples=100)
    for r, size, outer in results:
        print(f'Outer r{r}: {outer}, SP: {sp_diversity}')
