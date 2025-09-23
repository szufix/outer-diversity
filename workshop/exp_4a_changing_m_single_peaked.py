import math

import matplotlib.pyplot as plt
import pandas as pd
import csv
import os
from diversity import compute_outer_diversity
from diversity.sampling import (
    outer_diversity_sampling,
    outer_diversity_sampling_for_structered_domains
)
from src.domain.single_peaked import single_peaked_domain
from src.domain.group_separable import (
    group_separable_caterpillar_domain,
    group_separable_balanced_domain
)
from src.optimal_votes import find_optimal_facilities_sampled_simulated_annealing


def normalization(m):
    return math.factorial(m) // 2 * math.comb(m, 2)

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
    # Use sampled simulated annealing to find optimal votes directly
    optimal_votes, outer_diversity = find_optimal_facilities_sampled_simulated_annealing(
        num_candidates,
        domain_size,
        max_iterations=max_iterations,
        num_samples=num_samples,
        start_with='ic')

    return outer_diversity


def compute_diversity_comparison_data(
        candidate_range, num_samples, max_iterations, with_max=True):
    """Compute diversity data and export to CSV."""
    sp_diversities = []
    gs_caterpillar_diversities = []
    gs_balanced_diversities = []

    optimal_diversities = []
    domain_sizes = []

    print("Computing diversities for different numbers of candidates...")

    for num_candidates in candidate_range:
        print(f"Processing {num_candidates} candidates...")
        sp_diversity, sp_size = compute_single_peaked_diversity(
            num_candidates, num_samples)
        sp_diversities.append(sp_diversity)
        domain_sizes.append(sp_size)

        gs_caterpillar_diversity, _ = compute_group_separable_caterpillar_diversity(
            num_candidates, num_samples)
        gs_caterpillar_diversities.append(gs_caterpillar_diversity)
        gs_balanced_diversity, _ = compute_group_separable_balanced_diversity(
            num_candidates, num_samples
        )
        gs_balanced_diversities.append(gs_balanced_diversity)

        if with_max:
            optimal_diversity = compute_optimal_diversity_for_size(
                num_candidates, sp_size, num_samples, max_iterations)
            optimal_diversities.append(optimal_diversity)

    # Export data to CSV for record-keeping

    results_dir = os.path.join(os.path.dirname(__file__), 'data', 'changing_num_candidates')
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, 'diversity_comparison.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['num_candidates',
                      'sp_diversity',
                      'gs_caterpillar_diversity',
                      'gs_balanced_diversity',
                      'optimal_diversity',
                       'domain_size']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i, num_candidates in enumerate(candidate_range):
            row = {
                'num_candidates': num_candidates,
                'sp_diversity': sp_diversities[i],
                'gs_caterpillar_diversity': gs_caterpillar_diversities[i],
                'gs_balanced_diversity': gs_balanced_diversities[i],
                'domain_size': domain_sizes[i],
                'optimal_diversity': optimal_diversities[i] if with_max else None
            }
            writer.writerow(row)

def plot_diversity_comparison(with_max=True):
    """Plot diversity comparison from CSV."""
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    results_dir = os.path.join(os.path.dirname(__file__), 'data', 'changing_num_candidates')
    csv_path = os.path.join(results_dir, 'diversity_comparison.csv')
    df = pd.read_csv(csv_path)
    candidate_range = df['num_candidates']
    sp_diversities = df['sp_diversity']
    gs_caterpillar_diversities = df['gs_caterpillar_diversity']
    gs_balanced_diversities = df['gs_balanced_diversity']
    optimal_diversities = df['optimal_diversity'] if with_max else None

    plt.figure(figsize=(10, 6))
    if with_max and optimal_diversities is not None:
        plt.plot(candidate_range, optimal_diversities,
                 label='~Maximum Possible (Simulated Annealing)',
                 marker='s',
                 linewidth=2,
                 markersize=8,
                 color='tab:green')

    plt.plot(candidate_range, gs_balanced_diversities,
             label='GS Balanced Domain',
             marker='o',
             linewidth=2,
             markersize=8,
             color='tab:purple')

    plt.plot(candidate_range, gs_caterpillar_diversities,
             label='GS Caterpillar Domain',
             marker='o',
             linewidth=2,
             markersize=8,
             color='tab:blue')

    plt.plot(candidate_range, sp_diversities,
             label='Single-Peaked Domain',
             marker='o',
             linewidth=2,
             markersize=8,
             color='tab:red')




    plt.xlabel('Number of Candidates', fontsize=22)
    plt.ylabel('Outer Diversity Score', fontsize=22)
    if with_max:
        plt.title('Outer Diversity: Single-Peaked vs Maximum Possible', fontsize=20)
    else:
        plt.title('Outer Diversity: Single-Peaked', fontsize=20)
    plt.legend(fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.xticks(candidate_range, fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('img/changing_num_candidates_diversity_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def import_diversity_comparison_results():
    """Import diversity comparison results from CSV file."""
    import os
    import pandas as pd
    results_dir = os.path.join(os.path.dirname(__file__), 'data', 'changing_num_candidates')
    csv_path = os.path.join(results_dir, 'diversity_comparison.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Results file not found: {csv_path}")
    return pd.read_csv(csv_path)



if __name__ == "__main__":
    candidate_range = range(2, 9+1)  # 3 to 10 candidates
    num_samples = 1000
    max_iterations = 1
    compute_diversity_comparison_data(
        candidate_range, num_samples, max_iterations, with_max=False)
    plot_diversity_comparison(with_max=False)
