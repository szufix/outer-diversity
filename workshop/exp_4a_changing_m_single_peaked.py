from time import time
import csv
import os
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
import threading


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
    fieldnames = ['run', 'num_candidates', 'sp_diversity', 'gs_caterpillar_diversity', 'gs_balanced_diversity', 'optimal_diversity', 'domain_size']
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for run in range(num_runs):
            sp_diversities = []
            gs_caterpillar_diversities = []
            gs_balanced_diversities = []
            optimal_diversities = []
            domain_sizes = []
            print(f"Run {run+1}/{num_runs}")
            for num_candidates in candidate_range:
                print(f"Processing {num_candidates} candidates...")
                sp_diversity, sp_size = compute_single_peaked_diversity(num_candidates, num_samples)
                gs_caterpillar_diversity, _ = compute_group_separable_caterpillar_diversity(num_candidates, num_samples)
                gs_balanced_diversity, _ = compute_group_separable_balanced_diversity(num_candidates, num_samples)
                sp_diversities.append(sp_diversity)
                gs_caterpillar_diversities.append(gs_caterpillar_diversity)
                gs_balanced_diversities.append(gs_balanced_diversity)
                domain_sizes.append(sp_size)
                if with_max:
                    optimal_diversity = compute_optimal_diversity_for_size(num_candidates, sp_size, num_samples, max_iterations)
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

def compute_diversity_comparison_data_for_candidate(num_candidates, num_samples, max_iterations, with_max=True, num_runs=5, results_dir=None):
    """
    Compute diversity data for num_runs runs for a single num_candidates value and export results to a separate CSV.
    """
    import os
    if results_dir is None:
        results_dir = os.path.join(os.path.dirname(__file__), 'data', 'changing_m')
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f'_single_peaked_joint_{num_candidates}.csv')
    fieldnames = ['run', 'num_candidates', 'sp_diversity', 'gs_caterpillar_diversity', 'gs_balanced_diversity', 'optimal_diversity', 'domain_size']
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for run in range(num_runs):
            print(f"Run {run+1}/{num_runs} for {num_candidates} candidates...")
            sp_diversity, sp_size = compute_single_peaked_diversity(num_candidates, num_samples)
            gs_caterpillar_diversity, _ = compute_group_separable_caterpillar_diversity(num_candidates, num_samples)
            gs_balanced_diversity, _ = compute_group_separable_balanced_diversity(num_candidates, num_samples)
            if with_max:
                optimal_diversity = compute_optimal_diversity_for_size(num_candidates, sp_size, num_samples, max_iterations)
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

def compute_diversity_comparison_data_for_candidate_run(num_candidates, run, num_samples, max_iterations, with_max=True, results_dir=None):
    """
    Compute diversity data for a single run and num_candidates value, export to a separate CSV.
    """
    import os
    if results_dir is None:
        results_dir = os.path.join(os.path.dirname(__file__), 'data', 'changing_m')
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, 'single_peaked', f'_single_peaked_{num_candidates}_run{run}.csv')
    fieldnames = ['run', 'num_candidates', 'sp_diversity', 'gs_caterpillar_diversity', 'gs_balanced_diversity', 'optimal_diversity', 'domain_size']
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        sp_diversity, sp_size = compute_single_peaked_diversity(num_candidates, num_samples)
        gs_caterpillar_diversity, _ = compute_group_separable_caterpillar_diversity(num_candidates, num_samples)
        gs_balanced_diversity, _ = compute_group_separable_balanced_diversity(num_candidates, num_samples)
        if with_max:
            optimal_diversity = compute_optimal_diversity_for_size(num_candidates, sp_size, num_samples, max_iterations)
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

def run_fully_parallel_diversity_computation(candidate_range, num_samples, max_iterations, with_max=True, num_runs=5):
    """
    Run diversity computation in parallel threads for each (num_candidates, run) pair.
    """
    threads = []
    results_dir = os.path.join(os.path.dirname(__file__), 'data', 'changing_m')
    for num_candidates in candidate_range:
        for run in range(num_runs):
            t = threading.Thread(target=compute_diversity_comparison_data_for_candidate_run,
                                 args=(num_candidates, run, num_samples, max_iterations, with_max, results_dir))
            t.start()
            threads.append(t)
    for t in threads:
        t.join()

def plot_joint_diversity_comparison(with_max=True):
    """
    Plot diversity comparison from joint CSV, showing mean and std across all runs.
    Also print the mean and std for each candidate count.
    """
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
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
    plt.figure(figsize=(10, 6))
    plt.plot(candidate_range, sp_mean, label='Single-Peaked Domain', marker='o', linewidth=2, markersize=8, color='tab:blue')
    plt.fill_between(candidate_range, sp_mean - sp_std, sp_mean + sp_std, color='tab:blue', alpha=0.2)
    plt.plot(candidate_range, gs_caterpillar_mean, label='Group-Separable Caterpillar', marker='^', linewidth=2, markersize=8, color='tab:orange')
    plt.fill_between(candidate_range, gs_caterpillar_mean - gs_caterpillar_std, gs_caterpillar_mean + gs_caterpillar_std, color='tab:orange', alpha=0.2)
    plt.plot(candidate_range, gs_balanced_mean, label='Group-Separable Balanced', marker='s', linewidth=2, markersize=8, color='tab:purple')
    plt.fill_between(candidate_range, gs_balanced_mean - gs_balanced_std, gs_balanced_mean + gs_balanced_std, color='tab:purple', alpha=0.2)
    if with_max:
        plt.plot(candidate_range, opt_mean, label='~Maximum Possible (Simulated Annealing)', marker='x', linewidth=2, markersize=8, color='tab:green')
        plt.fill_between(candidate_range, opt_mean - opt_std, opt_mean + opt_std, color='tab:green', alpha=0.2)
    plt.xlabel('Number of Candidates', fontsize=22)
    plt.ylabel('Outer Diversity Score', fontsize=22)
    plt.title('Outer Diversity: Single-Peaked, Group-Separable, Maximum (Mean ± Std)', fontsize=20)
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(candidate_range, fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('images/changing_m_single_peaked_joint.png', dpi=300, bbox_inches='tight')
    plt.show()

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
    # plot_joint_diversity_comparison(with_max=True)
