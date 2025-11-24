import csv
import glob
import math
import multiprocessing
import os
import re

import pandas as pd

from src.diversity.sampling import (
    outer_diversity_sampling_for_structered_domains, outer_diversity_sampling
)
from src.domain.euclidean_ilp import euclidean_3d_domain, euclidean_2d_domain
from src.domain.single_peaked import spoc_domain
from src.domain.single_crossing import single_crossing_domain

from src.diversity.sampling import (
    outer_diversity_sampling,
    outer_diversity_sampling_for_structered_domains
)
from src.domain.group_separable import (
    group_separable_caterpillar_domain,
    group_separable_balanced_domain
)
from src.domain.single_peaked import single_peaked_domain


def normalization(m):
    return math.factorial(m) // 2 * math.comb(m, 2)



def compute_single_peaked_diversity(num_candidates, num_samples, inner_distance):
    """Compute outer diversity for single-peaked domain."""
    domain = single_peaked_domain(num_candidates)

    if inner_distance == 'swap':
        # FAST APPROACH
        outer_diversity, num_votes = outer_diversity_sampling_for_structered_domains(
            'sp', domain, num_candidates=num_candidates, num_samples=num_samples)
    else:
        # STANDARD APPROACH
        outer_diversity, num_votes = outer_diversity_sampling(
            domain, num_samples=num_samples)

    return outer_diversity, len(domain)


def compute_group_separable_caterpillar_diversity(num_candidates, num_samples, inner_distance):
    """Compute outer diversity for single-peaked domain."""
    domain = group_separable_caterpillar_domain(num_candidates)

    # STANDARD APPROACH
    outer_diversity, num_votes = outer_diversity_sampling(
        domain, num_samples=num_samples)

    return outer_diversity, len(domain)


def compute_group_separable_balanced_diversity(num_candidates, num_samples, inner_distance):
    """Compute outer diversity for single-peaked domain."""
    domain = group_separable_balanced_domain(num_candidates)

    # STANDARD APPROACH
    outer_diversity, num_votes = outer_diversity_sampling(
        domain, num_samples=num_samples)

    return outer_diversity, len(domain)




def compute_single_crossing_diversity(num_candidates, num_samples, inner_distance):
    """Compute outer diversity for single-peaked domain."""
    domain = single_crossing_domain(num_candidates)

    if inner_distance == 'swap':
        # FAST APPROACH
        outer_diversity, num_votes = outer_diversity_sampling_for_structered_domains(
        'sc', domain, num_candidates=num_candidates, num_samples=num_samples)
    else:
        # STANDARD APPROACH
        outer_diversity, num_votes = outer_diversity_sampling(
            domain, num_samples=num_samples, inner_distance=inner_distance)

    return outer_diversity, len(domain)


def compute_euclidean_3d_diversity(num_candidates, num_samples, inner_distance):
    """Compute outer diversity for 2D domain."""
    domain = euclidean_3d_domain(num_candidates)

    # STANDARD APPROACH
    outer_diversity, num_votes = outer_diversity_sampling(
        domain, num_samples=num_samples, inner_distance=inner_distance)

    return outer_diversity, len(domain)


def compute_spoc_diversity(num_candidates, num_samples, inner_distance):
    """Compute outer diversity for 2D domain."""
    domain = spoc_domain(num_candidates)

    if inner_distance == 'swap':
        # FAST APPROACH
        outer_diversity, num_votes = outer_diversity_sampling_for_structered_domains(
            'spoc', domain, num_candidates=num_candidates, num_samples=num_samples)
    else:
        # STANDARD APPROACH
        outer_diversity, num_votes = outer_diversity_sampling(
            domain, num_samples=num_samples, inner_distance=inner_distance)


    return outer_diversity, len(domain)



def compute_euclidean_2d_diversity(num_candidates, num_samples, inner_distance='swap'):
    """Compute outer diversity for 2D domain."""
    domain = euclidean_2d_domain(num_candidates)

    # STANDARD APPROACH
    outer_diversity, num_votes = outer_diversity_sampling(
        domain, num_samples=num_samples, inner_distance=inner_distance)

    return outer_diversity, len(domain)

diversity_func = {
    'euclidean_2d': compute_euclidean_2d_diversity,
    'euclidean_3d': compute_euclidean_3d_diversity,
    'spoc': compute_spoc_diversity,
    'single_crossing': compute_single_crossing_diversity,
    'single_peaked': compute_single_peaked_diversity,
    'balanced': compute_group_separable_balanced_diversity,
    'caterpillar': compute_group_separable_caterpillar_diversity,
}


def compute_diversity_comparison_data_for_candidate_run(
        name, num_candidates, run, num_samples,  results_dir=None, inner_distance='swap'):
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

        diversity, domain_size = diversity_func[name](num_candidates, num_samples, inner_distance)

        row = {
            'run': run,
            'num_candidates': num_candidates,
            f'{name}_diversity': diversity,
            'domain_size': domain_size
        }
        writer.writerow(row)


def run_fully_parallel_diversity_computation(
        name,
        candidate_range,
        num_samples,
        runs_range,
        inner_distance,
):
    """
    Run diversity computation in parallel processes for each (num_candidates, run) pair.
    """
    processes = []
    results_dir = os.path.join(os.path.dirname(__file__), 'data', 'changing_m')
    for num_candidates in candidate_range:
        for run in runs_range:
            p = multiprocessing.Process(target=compute_diversity_comparison_data_for_candidate_run,
                                       args=(name, num_candidates, run, num_samples,
                                             results_dir, inner_distance))
            p.start()
            processes.append(p)
    for p in processes:
        p.join()


# def merge_results(name):
#     results_dir = os.path.join(os.path.dirname(__file__), 'data', 'changing_m', 'other')
#     output_path = os.path.join(os.path.dirname(__file__), 'data', 'changing_m', f'_{name}_joint.csv')
#     all_files = glob.glob(os.path.join(results_dir, f'_{name}_*_run*.csv'))
#     print(all_files)
#     dfs = [pd.read_csv(f) for f in all_files]
#     merged = pd.concat(dfs, ignore_index=True)
#     merged.to_csv(output_path, index=False)
#     print(f"Merged {len(all_files)} files into {output_path}")



def merge_results(name, candidate_range, runs_range):
    results_dir = os.path.join(os.path.dirname(__file__), 'data', 'changing_m', 'other')
    output_path = os.path.join(os.path.dirname(__file__), 'data', 'changing_m', f'_{name}_joint.csv')
    all_files = glob.glob(os.path.join(results_dir, f'_{name}_*_run*.csv'))

    filtered_files = []
    for f in all_files:
        match = re.search(fr'_{name}_(\d+)_run(\d+)\.csv', os.path.basename(f))
        if match:
            candidate = int(match.group(1))
            run = int(match.group(2))
            if candidate in candidate_range and run in runs_range:
                filtered_files.append(f)

    dfs = [pd.read_csv(f) for f in filtered_files]
    if dfs:
        merged = pd.concat(dfs, ignore_index=True)
        merged.to_csv(output_path, index=False)
        print(f"Merged {len(filtered_files)} files into {output_path}")
    else:
        print("No files matched the given candidate and run ranges.")



if __name__ == "__main__":

    names = [
        'single_peaked',
        'single_crossing',
        'euclidean_2d',
        'euclidean_3d',
        'spoc',
        'balanced',
        'caterpillar'
    ]
    candidate_range = range(2, 8+1)
    num_samples = 1000
    runs_range = range(10)
    inner_distance = 'spearman'

    for name in names:
        run_fully_parallel_diversity_computation(
            name, candidate_range, num_samples, runs_range, inner_distance)
        merge_results(name, candidate_range, runs_range)