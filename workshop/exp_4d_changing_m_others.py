import csv
import glob
import math
import os
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import multiprocessing

from src.diversity.sampling import (
    outer_diversity_sampling_for_structered_domains, outer_diversity_sampling
)
from src.max_diversity.main import find_optimal_facilities_sampled_simulated_annealing
from src.domain.euclidean_ilp import euclidean_3d_domain
from src.print_utils import LABEL, MARKER, COLOR, LINE
from src.domain.single_peaked import spoc_domain


def normalization(m):
    return math.factorial(m) // 2 * math.comb(m, 2)



def compute_euclidean_3d_diversity(num_candidates, num_samples):
    """Compute outer diversity for 2D domain."""
    domain = euclidean_3d_domain(num_candidates)

    # STANDARD APPROACH
    outer_diversity, num_votes = outer_diversity_sampling(
        domain, num_samples=num_samples)

    return outer_diversity, len(domain)


def compute_spoc_diversity(num_candidates, num_samples):
    """Compute outer diversity for 2D domain."""
    domain = spoc_domain(num_candidates)

    # FAST APPROACH
    outer_diversity, num_votes = outer_diversity_sampling_for_structered_domains(
        'spoc', domain, num_candidates=num_candidates, num_samples=num_samples)

    # STANDARD APPROACH
    # outer_diversity, num_votes = outer_diversity_sampling(
    #     domain, num_samples=num_samples)


    return outer_diversity, len(domain)


diversity_func = {
    'euclidean_3d': compute_euclidean_3d_diversity,
    'spoc': compute_spoc_diversity,
}


def compute_diversity_comparison_data_for_candidate_run(
        name, num_candidates, run, num_samples,  results_dir=None):
    """
    Compute diversity data for a single run and num_candidates value, export to a separate CSV.
    """
    import os
    if results_dir is None:
        results_dir = os.path.join(os.path.dirname(__file__), 'data', 'changing_m')
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, 'other', f'_{name}_{num_candidates}_run{run}.csv')
    fieldnames = ['run', 'num_candidates', f'{name}_diversity', 'domain_size']
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        diversity, domain_size = diversity_func[name](num_candidates, num_samples)

        row = {
            'run': run,
            'num_candidates': num_candidates,
            f'{name}_diversity': diversity,
            'domain_size': domain_size
        }
        writer.writerow(row)


def run_fully_parallel_diversity_computation(name, candidate_range, num_samples, runs_range):
    """
    Run diversity computation in parallel processes for each (num_candidates, run) pair.
    """
    processes = []
    results_dir = os.path.join(os.path.dirname(__file__), 'data', 'changing_m')
    for num_candidates in candidate_range:
        for run in runs_range:
            p = multiprocessing.Process(target=compute_diversity_comparison_data_for_candidate_run,
                                       args=(name, num_candidates, run, num_samples, results_dir))
            p.start()
            processes.append(p)
    for p in processes:
        p.join()


def merge_results(name):
    results_dir = os.path.join(os.path.dirname(__file__), 'data', 'changing_m', 'other')
    output_path = os.path.join(os.path.dirname(__file__), 'data', 'changing_m', f'_{name}_joint.csv')
    all_files = glob.glob(os.path.join(results_dir, f'_{name}_*_run*.csv'))
    print(all_files)
    dfs = [pd.read_csv(f) for f in all_files]
    merged = pd.concat(dfs, ignore_index=True)
    merged.to_csv(output_path, index=False)
    print(f"Merged {len(all_files)} files into {output_path}")



if __name__ == "__main__":

    names = ['spoc', 'euclidean_3d']
    candidate_range = range(2, 20+1)
    num_samples = 1000
    runs_range = [1]
    for name in names:
        run_fully_parallel_diversity_computation(
            name, candidate_range, num_samples, runs_range)

        # merge_results(name)