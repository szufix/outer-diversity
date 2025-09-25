import csv
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
from src.domain.euclidean_ilp import euclidean_2d_domain

def normalization(m):
    return math.factorial(m) // 2 * math.comb(m, 2)


def compute_2d_diversity(num_candidates, num_samples):
    """Compute outer diversity for 2D domain."""
    domain = euclidean_2d_domain(num_candidates)

    # STANDARD APPROACH
    outer_diversity, num_votes = outer_diversity_sampling(
        domain, num_samples=num_samples)

    return outer_diversity, len(domain)



def compute_optimal_diversity_for_size(num_candidates, domain_size, num_samples, max_iterations):
    """Compute maximum possible outer diversity for a given domain size using sampled SA."""
    # Use sampled simulated annealing to find optimal votes directly
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
    Each row contains: run, num_candidates, twod_diversity, optimal_diversity, domain_size
    """
    results_dir = os.path.join(os.path.dirname(__file__), 'data', 'changing_m')
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, '_2d_joint.csv')
    fieldnames = ['run', 'num_candidates', '2d_diversity', 'optimal_diversity', 'domain_size']
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for run in range(num_runs):
            twod_diversities = []
            optimal_diversities = []
            domain_sizes = []
            print(f"Run {run+1}/{num_runs}")
            for num_candidates in candidate_range:
                print(f"Processing {num_candidates} candidates...")
                twod_diversity, twod_size = compute_2d_diversity(num_candidates, num_samples)
                twod_diversities.append(twod_diversity)
                domain_sizes.append(twod_size)
                if with_max:
                    optimal_diversity = compute_optimal_diversity_for_size(num_candidates, twod_size, num_samples, max_iterations)
                    optimal_diversities.append(optimal_diversity)
            for i, num_candidates in enumerate(candidate_range):
                row = {
                    'run': run,
                    'num_candidates': num_candidates,
                    '2d_diversity': twod_diversities[i],
                    'domain_size': domain_sizes[i],
                    'optimal_diversity': optimal_diversities[i] if with_max else None
                }
                writer.writerow(row)

def plot_diversity_comparison(with_max=True):
    """Plot diversity comparison from CSV."""

    results_dir = os.path.join(os.path.dirname(__file__), 'data', 'changing_m')
    csv_path = os.path.join(results_dir, '_2d.csv')
    df = pd.read_csv(csv_path)
    candidate_range = df['num_candidates']
    twod_diversities = df['2d_diversity']
    optimal_diversities = df['optimal_diversity'] if with_max else None

    plt.figure(figsize=(10, 6))
    if with_max and optimal_diversities is not None:
        plt.plot(candidate_range, optimal_diversities,
                 label='~Maximum Possible (Simulated Annealing)',
                 marker='s',
                 linewidth=2,
                 markersize=8,
                 color='tab:green')

    plt.plot(candidate_range, twod_diversities,
             label='2D Domain',
             marker='o',
             linewidth=2,
             markersize=8,
             color='tab:red')

    plt.xlabel('Number of Candidates', fontsize=22)
    plt.ylabel('Outer Diversity Score', fontsize=22)
    if with_max:
        plt.title('Outer Diversity: 2D vs Maximum Possible', fontsize=20)
    else:
        plt.title('Outer Diversity: 2D', fontsize=20)
    plt.legend(fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.xticks(candidate_range, fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('images/changing_m_2d.png', dpi=300, bbox_inches='tight')
    plt.show()


def import_diversity_comparison_results():
    """Import diversity comparison results from CSV file."""
    import os
    import pandas as pd
    results_dir = os.path.join(os.path.dirname(__file__), 'data', 'changing_m')
    csv_path = os.path.join(results_dir, '_2d.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Results file not found: {csv_path}")
    return pd.read_csv(csv_path)

def plot_joint_diversity_comparison(with_max=True):
    """
    Plot diversity comparison from joint CSV, showing mean and std across all runs.
    Also print the mean and std for each candidate count.
    """

    results_dir = os.path.join(os.path.dirname(__file__), 'data', 'changing_m')
    csv_path = os.path.join(results_dir, '_2d_joint.csv')
    df = pd.read_csv(csv_path)
    grouped = df.groupby('num_candidates')
    candidate_range = np.array(sorted(df['num_candidates'].unique()))
    twod_mean = grouped['2d_diversity'].mean().values
    twod_std = grouped['2d_diversity'].std().values
    if with_max:
        opt_mean = grouped['optimal_diversity'].mean().values
        opt_std = grouped['optimal_diversity'].std().values
    # Print table of means and stds
    # Plot
    plt.figure(figsize=(10, 6))
    if with_max:
        plt.plot(candidate_range, opt_mean, label='~Maximum Possible (Simulated Annealing)', marker='s', linewidth=2, markersize=8, color='tab:green')
        plt.fill_between(candidate_range, opt_mean - opt_std, opt_mean + opt_std, color='tab:green', alpha=0.2)
    plt.plot(candidate_range, twod_mean, label='2D Domain', marker='o', linewidth=2, markersize=8, color='tab:red')
    plt.fill_between(candidate_range, twod_mean - twod_std, twod_mean + twod_std, color='tab:red', alpha=0.2)
    plt.xlabel('Number of Candidates', fontsize=22)
    plt.ylabel('Outer Diversity Score', fontsize=22)
    if with_max:
        plt.title('Outer Diversity: 2D vs Maximum Possible', fontsize=20)
    else:
        plt.title('Outer Diversity: 2D', fontsize=20)
    plt.legend(fontsize=16)
    plt.grid(True, alpha=0.3)
    # plt.xticks(candidate_range, fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('images/changing_m_2d_joint.png', dpi=300, bbox_inches='tight')
    plt.show()


def compute_diversity_comparison_data_for_candidate_run(num_candidates, run, num_samples, max_iterations, with_max=True, results_dir=None):
    """
    Compute diversity data for a single run and num_candidates value, export to a separate CSV.
    """
    import os
    if results_dir is None:
        results_dir = os.path.join(os.path.dirname(__file__), 'data', 'changing_m')
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, 'euclidean_2d', f'_2d_{num_candidates}_run{run}.csv')
    fieldnames = ['run', 'num_candidates', '2d_diversity', 'optimal_diversity', 'domain_size']
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        twod_diversity, twod_size = compute_2d_diversity(num_candidates, num_samples)
        if with_max:
            optimal_diversity = compute_optimal_diversity_for_size(num_candidates, twod_size, num_samples, max_iterations)
        else:
            optimal_diversity = None
        row = {
            'run': run,
            'num_candidates': num_candidates,
            '2d_diversity': twod_diversity,
            'domain_size': twod_size,
            'optimal_diversity': optimal_diversity
        }
        writer.writerow(row)

def run_fully_parallel_diversity_computation(candidate_range, num_samples, max_iterations, with_max=True, num_runs=5):
    """
    Run diversity computation in parallel processes for each (num_candidates, run) pair.
    """
    processes = []
    results_dir = os.path.join(os.path.dirname(__file__), 'data', 'changing_m')
    for num_candidates in candidate_range:
        for run in range(num_runs):
            p = multiprocessing.Process(target=compute_diversity_comparison_data_for_candidate_run,
                                       args=(num_candidates, run, num_samples, max_iterations, with_max, results_dir))
            p.start()
            processes.append(p)
    for p in processes:
        p.join()

if __name__ == "__main__":
    candidate_range = range(2, 20+1)
    num_samples = 1000
    max_iterations = 256
    num_runs = 10
    start_time = time()
    run_fully_parallel_diversity_computation(
        candidate_range, num_samples, max_iterations, with_max=True, num_runs=num_runs)
    end_time = time()
    print(f"Computation time: {end_time - start_time} seconds")
    # plot_joint_diversity_comparison(with_max=with_max)
