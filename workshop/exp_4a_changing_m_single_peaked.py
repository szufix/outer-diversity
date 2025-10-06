import glob
from time import time
import csv
import os

import pandas as pd

from src.diversity.sampling import (
    outer_diversity_sampling,
    outer_diversity_sampling_for_structered_domains
)
from src.domain.single_peaked import single_peaked_domain
from src.domain.group_separable import (
    group_separable_caterpillar_domain,
    group_separable_balanced_domain
)
from src.max_diversity.main import find_optimal_facilities_sampled_simulated_annealing
import multiprocessing
from src.print_utils import LABEL, MARKER, COLOR, LINE
import re
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def compute_single_peaked_diversity(num_candidates, num_samples):
    """Compute outer diversity for single-peaked domain."""
    domain = single_peaked_domain(num_candidates)

    # FAST APPROACH
    outer_diversity, num_votes = outer_diversity_sampling_for_structered_domains(
        'sp', domain, num_candidates=num_candidates, num_samples=num_samples)

    # STANDARD APPROACH
    # outer_diversity, num_votes = outer_diversity_sampling(
    #     domain, num_samples=num_samples)

    return outer_diversity, len(domain)


def compute_group_separable_caterpillar_diversity(num_candidates, num_samples):
    """Compute outer diversity for single-peaked domain."""
    domain = group_separable_caterpillar_domain(num_candidates)

    # STANDARD APPROACH
    outer_diversity, num_votes = outer_diversity_sampling(
        domain, num_samples=num_samples)

    return outer_diversity, len(domain)


def compute_group_separable_balanced_diversity(num_candidates, num_samples):
    """Compute outer diversity for single-peaked domain."""
    domain = group_separable_balanced_domain(num_candidates)

    # STANDARD APPROACH
    outer_diversity, num_votes = outer_diversity_sampling(
        domain, num_samples=num_samples)

    return outer_diversity, len(domain)


def compute_optimal_diversity_for_size(num_candidates, domain_size, num_samples, max_iterations):
    """Compute maximum possible outer diversity for a given domain size using sampled SA."""
    optimal_votes, outer_diversity = find_optimal_facilities_sampled_simulated_annealing(
        num_candidates,
        domain_size,
        max_iterations=max_iterations,
        num_samples=num_samples,
        start_with='ic')
    return outer_diversity


def compute_diversity_comparison_data(
        candidate_range, num_samples, max_iterations, with_max=True, num_runs=5):
    """
    Compute diversity data for num_runs runs and export all results to a joint CSV.
    Each row contains: run, num_candidates, sp_diversity, gs_caterpillar_diversity, gs_balanced_diversity, optimal_diversity, domain_size
    """
    results_dir = os.path.join(os.path.dirname(__file__), 'data', 'changing_m')
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, '_single_peaked_joint.csv')
    fieldnames = ['run', 'num_candidates', 'sp_diversity', 'gs_caterpillar_diversity',
                  'gs_balanced_diversity', 'optimal_diversity', 'domain_size']
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for run in range(num_runs):
            sp_diversities = []
            gs_caterpillar_diversities = []
            gs_balanced_diversities = []
            optimal_diversities = []
            domain_sizes = []
            print(f"Run {run + 1}/{num_runs}")
            for num_candidates in candidate_range:
                print(f"Processing {num_candidates} candidates...")
                sp_diversity, sp_size = compute_single_peaked_diversity(num_candidates, num_samples)
                gs_caterpillar_diversity, _ = compute_group_separable_caterpillar_diversity(
                    num_candidates, num_samples)
                gs_balanced_diversity, _ = compute_group_separable_balanced_diversity(
                    num_candidates, num_samples)
                sp_diversities.append(sp_diversity)
                gs_caterpillar_diversities.append(gs_caterpillar_diversity)
                gs_balanced_diversities.append(gs_balanced_diversity)
                domain_sizes.append(sp_size)
                if with_max:
                    optimal_diversity = compute_optimal_diversity_for_size(num_candidates, sp_size,
                                                                           num_samples,
                                                                           max_iterations)
                    optimal_diversities.append(optimal_diversity)
            for i, num_candidates in enumerate(candidate_range):
                row = {
                    'run': run,
                    'num_candidates': num_candidates,
                    'sp_diversity': sp_diversities[i],
                    'gs_caterpillar_diversity': gs_caterpillar_diversities[i],
                    'gs_balanced_diversity': gs_balanced_diversities[i],
                    'domain_size': domain_sizes[i],
                    'optimal_diversity': optimal_diversities[i] if with_max else None
                }
                writer.writerow(row)


def compute_diversity_comparison_data_for_candidate(num_candidates, num_samples, max_iterations,
                                                    with_max=True, num_runs=5, results_dir=None):
    """
    Compute diversity data for num_runs runs for a single num_candidates value and export results to a separate CSV.
    """
    import os
    if results_dir is None:
        results_dir = os.path.join(os.path.dirname(__file__), 'data', 'changing_m')
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f'_single_peaked_joint_{num_candidates}.csv')
    fieldnames = ['run', 'num_candidates', 'sp_diversity', 'gs_caterpillar_diversity',
                  'gs_balanced_diversity', 'optimal_diversity', 'domain_size']
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for run in range(num_runs):
            print(f"Run {run + 1}/{num_runs} for {num_candidates} candidates...")
            sp_diversity, sp_size = compute_single_peaked_diversity(num_candidates, num_samples)
            gs_caterpillar_diversity, _ = compute_group_separable_caterpillar_diversity(
                num_candidates, num_samples)
            gs_balanced_diversity, _ = compute_group_separable_balanced_diversity(num_candidates,
                                                                                  num_samples)
            if with_max:
                optimal_diversity = compute_optimal_diversity_for_size(num_candidates, sp_size,
                                                                       num_samples, max_iterations)
            else:
                optimal_diversity = None
            row = {
                'run': run,
                'num_candidates': num_candidates,
                'sp_diversity': sp_diversity,
                'gs_caterpillar_diversity': gs_caterpillar_diversity,
                'gs_balanced_diversity': gs_balanced_diversity,
                'domain_size': sp_size,
                'optimal_diversity': optimal_diversity
            }
            writer.writerow(row)


def compute_diversity_comparison_data_for_candidate_run(num_candidates, run, num_samples,
                                                        max_iterations, with_max=True,
                                                        results_dir=None):
    """
    Compute diversity data for a single run and num_candidates value, export to a separate CSV.
    """
    import os
    if results_dir is None:
        results_dir = os.path.join(os.path.dirname(__file__), 'data', 'changing_m')
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, 'single_peaked',
                            f'_single_peaked_{num_candidates}_run{run}.csv')
    fieldnames = ['run', 'num_candidates', 'sp_diversity', 'gs_caterpillar_diversity',
                  'gs_balanced_diversity', 'optimal_diversity', 'domain_size']
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        sp_diversity, sp_size = compute_single_peaked_diversity(num_candidates, num_samples)
        gs_caterpillar_diversity, _ = compute_group_separable_caterpillar_diversity(num_candidates,
                                                                                    num_samples)
        gs_balanced_diversity, _ = compute_group_separable_balanced_diversity(num_candidates,
                                                                              num_samples)
        if with_max:
            optimal_diversity = compute_optimal_diversity_for_size(num_candidates, sp_size,
                                                                   num_samples, max_iterations)
        else:
            optimal_diversity = None
        row = {
            'run': run,
            'num_candidates': num_candidates,
            'sp_diversity': sp_diversity,
            'gs_caterpillar_diversity': gs_caterpillar_diversity,
            'gs_balanced_diversity': gs_balanced_diversity,
            'domain_size': sp_size,
            'optimal_diversity': optimal_diversity
        }
        writer.writerow(row)


def run_fully_parallel_diversity_computation(candidate_range, num_samples, max_iterations,
                                             with_max=True, num_runs=5):
    """
    Run diversity computation in parallel processes for each (num_candidates, run) pair.
    """
    processes = []
    results_dir = os.path.join(os.path.dirname(__file__), 'data', 'changing_m')
    for num_candidates in candidate_range:
        for run in range(num_runs):
            p = multiprocessing.Process(target=compute_diversity_comparison_data_for_candidate_run,
                                        args=(num_candidates, run, num_samples, max_iterations,
                                              with_max, results_dir))
            p.start()
            processes.append(p)
    for p in processes:
        p.join()


def plot_joint_diversity_comparison(with_max=True):
    """
    Plot diversity comparison from joint CSV, showing mean and std across all runs.
    Also print the mean and std for each candidate count.
    """

    results_dir = os.path.join(os.path.dirname(__file__), 'data', 'changing_m')
    csv_path = os.path.join(results_dir, '_single_peaked_joint.csv')
    df = pd.read_csv(csv_path)
    grouped = df.groupby('num_candidates')
    candidate_range = np.array(sorted(df['num_candidates'].unique()))
    sp_mean = grouped['sp_diversity'].mean().values
    sp_std = grouped['sp_diversity'].std().values
    gs_caterpillar_mean = grouped['gs_caterpillar_diversity'].mean().values
    gs_caterpillar_std = grouped['gs_caterpillar_diversity'].std().values
    gs_balanced_mean = grouped['gs_balanced_diversity'].mean().values
    gs_balanced_std = grouped['gs_balanced_diversity'].std().values
    if with_max:
        opt_mean = grouped['optimal_diversity'].mean().values
        opt_std = grouped['optimal_diversity'].std().values
    # Plot

    plt.figure(figsize=(6, 6))
    if with_max:
        plt.plot(candidate_range, opt_mean, label=LABEL['max'], marker=MARKER['max'], linewidth=2,
                 markersize=8, color=COLOR['max'], linestyle=LINE['max'])
        plt.fill_between(candidate_range, opt_mean - opt_std, opt_mean + opt_std,
                         color=COLOR['max'], alpha=0.2)

    plt.plot(candidate_range, gs_caterpillar_mean, label=LABEL['caterpillar'],
             marker=MARKER['caterpillar'], linewidth=2, markersize=8, color=COLOR['caterpillar'])
    plt.fill_between(candidate_range, gs_caterpillar_mean - gs_caterpillar_std,
                     gs_caterpillar_mean + gs_caterpillar_std, color=COLOR['caterpillar'],
                     alpha=0.2)
    plt.plot(candidate_range, gs_balanced_mean, label=LABEL['balanced'], marker=MARKER['balanced'],
             linewidth=2, markersize=8, color=COLOR['balanced'])
    plt.fill_between(candidate_range, gs_balanced_mean - gs_balanced_std,
                     gs_balanced_mean + gs_balanced_std, color=COLOR['balanced'], alpha=0.2)
    plt.plot(candidate_range, sp_mean, label=LABEL['single_peaked'], marker=MARKER['single_peaked'],
             linewidth=2, markersize=8, color=COLOR['single_peaked'])
    plt.fill_between(candidate_range, sp_mean - sp_std, sp_mean + sp_std,
                     color=COLOR['single_peaked'], alpha=0.2)

    plt.xlabel('Number of Candidates', fontsize=36)
    # plt.ylabel('Outer Diversity', fontsize=36)
    plt.legend(fontsize=28, loc='lower left')
    plt.grid(True, alpha=0.3)
    xticks_to_show = [2, 5, 8, 11, 14]
    plt.xticks(xticks_to_show, fontsize=28)

    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], ['', '', '','','','',], fontsize=28)

    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('images/changing_m/changing_m_single_peaked_with_max.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_joint_diversity_comparison_normalized():
    """
    Plot diversity comparison from joint CSV, showing mean and std across all runs.
    Also print the mean and std for each candidate count.
    """

    results_dir = os.path.join(os.path.dirname(__file__), 'data', 'changing_m')
    csv_path = os.path.join(results_dir, '_single_peaked_joint.csv')
    df = pd.read_csv(csv_path)
    grouped = df.groupby('num_candidates')
    candidate_range = np.array(sorted(df['num_candidates'].unique()))
    sp_mean = grouped['sp_diversity'].mean().values
    sp_std = grouped['sp_diversity'].std().values
    gs_caterpillar_mean = grouped['gs_caterpillar_diversity'].mean().values
    gs_caterpillar_std = grouped['gs_caterpillar_diversity'].std().values
    gs_balanced_mean = grouped['gs_balanced_diversity'].mean().values
    gs_balanced_std = grouped['gs_balanced_diversity'].std().values

    opt_mean = grouped['optimal_diversity'].mean().values
    opt_std = grouped['optimal_diversity'].std().values

    # Normalize by optimal diversity
    sp_mean = sp_mean / opt_mean
    sp_std = sp_std / opt_mean
    gs_caterpillar_mean = gs_caterpillar_mean / opt_mean
    gs_caterpillar_std = gs_caterpillar_std / opt_mean
    gs_balanced_mean = gs_balanced_mean / opt_mean
    gs_balanced_std = gs_balanced_std / opt_mean

    plt.figure(figsize=(9, 6))

    plt.plot(candidate_range, gs_caterpillar_mean, label=LABEL['caterpillar'],
             marker=MARKER['caterpillar'], linewidth=2, markersize=8,
             color=COLOR['caterpillar'])
    plt.fill_between(candidate_range, gs_caterpillar_mean - gs_caterpillar_std,
                     gs_caterpillar_mean + gs_caterpillar_std, color=COLOR['caterpillar'],
                     alpha=0.2)
    plt.plot(candidate_range, gs_balanced_mean, label=LABEL['balanced'],
             marker=MARKER['balanced'],
             linewidth=2, markersize=8, color=COLOR['balanced'])
    plt.fill_between(candidate_range, gs_balanced_mean - gs_balanced_std,
                     gs_balanced_mean + gs_balanced_std, color=COLOR['balanced'], alpha=0.2)
    plt.plot(candidate_range, sp_mean, label=LABEL['single_peaked'],
             marker=MARKER['single_peaked'],
             linewidth=2, markersize=8, color=COLOR['single_peaked'])
    plt.fill_between(candidate_range, sp_mean - sp_std, sp_mean + sp_std,
                     color=COLOR['single_peaked'], alpha=0.2)

    plt.xlabel('Number of Candidates', fontsize=36)
    plt.ylabel('Normalized Diversity', fontsize=36)
    plt.legend(fontsize=28, loc='lower left')
    plt.grid(True, alpha=0.3)
    xticks_to_show = [2, 5, 8, 11, 14]
    plt.xticks(xticks_to_show, fontsize=28)
    plt.yticks(fontsize=28)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('images/changing_m/changing_m_single_peaked_normalized.png', dpi=300,
                bbox_inches='tight')
    plt.show()


def merge_single_peaked_results(candidate_range, runs_range, with_max):
    results_dir = os.path.join(os.path.dirname(__file__), 'data', 'changing_m', 'single_peaked')
    if with_max:
        output_path = os.path.join(os.path.dirname(__file__), 'data', 'changing_m',
                                   '_single_peaked_joint.csv')
        all_files = glob.glob(os.path.join(results_dir, '_single_peaked_*_run*.csv'))
    else:
        output_path = os.path.join(os.path.dirname(__file__), 'data', 'changing_m',
                                   '_single_peaked_joint_no_max.csv')
        all_files = glob.glob(os.path.join(results_dir, '_single_peaked_*_run*_no_max.csv'))

    filtered_files = []
    for f in all_files:
        if with_max:
            match = re.search(r'_single_peaked_(\d+)_run(\d+)\.csv', os.path.basename(f))
        else:
            match = re.search(r'_single_peaked_(\d+)_run(\d+)_no_max\.csv', os.path.basename(f))

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
    candidate_range = range(2, 14 + 1)
    num_samples = 1000
    max_iterations = 256
    runs_range = range(10)
    # run_fully_parallel_diversity_computation(
    #     candidate_range, num_samples, max_iterations, with_max=True, num_runs=num_runs)
    # merge_single_peaked_results(candidate_range, runs_range, with_max=True)
    plot_joint_diversity_comparison(with_max=True)
    # plot_joint_diversity_comparison_normalized()

    # merge_single_peaked_results(range(2, 17 + 1), runs_range, with_max=False)
