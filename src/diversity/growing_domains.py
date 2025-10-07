import csv
import math
import os
from tqdm import tqdm

from src.domain.extenders import (
    extend_by_swaps
)
from src.domain import *

from src.diversity.diversity_utils import (
    get_max_num_swaps,
)

domains = {
    'euclidean_3d': euclidean_3d_domain,
    'euclidean_2d': euclidean_2d_domain,
    'euclidean_1d': euclidean_1d_domain,
    'caterpillar': group_separable_caterpillar_domain,
    'balanced': group_separable_balanced_domain,
    'single_peaked': single_peaked_domain,
    'single_crossing': single_crossing_domain,
    'sp_double_forked': sp_double_forked_domain,
    'spoc': spoc_domain,
    'ext_single_vote': ext_single_vote_domain,
    'single_vote': single_vote_domain,
    'largest_condorcet': largest_condorcet_domain,
}


def load_domain_size_csv(num_candidates):
    csv_filename = f'data/domain_size/domain_size_m{num_candidates}.csv'
    size_data = {}
    with open(csv_filename, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header

        for row in reader:
            domain_name = row[0]
            sizes = [int(size) for size in row[1:]]
            size_data[domain_name] = sizes
    return size_data



def save_domain_size_csv(size, num_candidates, run_idx=None):
    max_num_swaps = get_max_num_swaps(num_candidates)
    x_range = list(range(0, max_num_swaps + 1))
    os.makedirs('data', exist_ok=True)
    if run_idx is not None:
        csv_filename = f'data/domain_size/domain_size_m{num_candidates}_run{run_idx}.csv'
    else:
        csv_filename = f'data/domain_size/domain_size_m{num_candidates}.csv'
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['domain'] + list(x_range))
        for name in size:
            writer.writerow([name] + size[name])


def compute_domain_balls(base, num_candidates, num_runs=10) -> None:
    """Computes domain balls for given parameters and saves each run to a separate CSV."""
    for run_idx in range(num_runs):
        balls = {name: [] for name in base}
        for name in base:
            domain = domains[name](num_candidates=num_candidates)
            balls[name].append(len(domain))
            max_num_swaps = get_max_num_swaps(num_candidates)
            for _ in tqdm(range(1, max_num_swaps+1), desc=f'{name} swaps'):
                # print(balls[name][-1], math.factorial(num_candidates))

                if balls[name][-1] == math.factorial(num_candidates):
                    print(balls)
                    print("\nFor", num_candidates, " outcome is: ", len(balls[name])-1)
                    break

                if len(balls[name]) >= 2 and balls[name][-1] == balls[name][-2]:
                    balls[name].append(balls[name][-1])
                else:
                    domain = extend_by_swaps(domain, 1)
                    balls[name].append(len(domain))
        # save_domain_size_csv(balls, num_candidates, run_idx=run_idx)
