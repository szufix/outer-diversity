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
from src.domain.single_crossing import single_crossing_domain
from src.domain.group_separable import (
    group_separable_caterpillar_domain,
    group_separable_balanced_domain
)
from src.optimal_votes import find_optimal_facilities_sampled_simulated_annealing


def normalization(m):
    return math.factorial(m) // 2 * math.comb(m, 2)


def compute_single_crossing_diversity(num_candidates, num_samples):
    """Compute outer diversity for single-peaked domain."""
    domain = single_crossing_domain(num_candidates)

    # FAST APPROACH
    outer_diversity, num_votes = outer_diversity_sampling_for_structered_domains(
        'sc', domain, num_candidates=num_candidates, num_samples=num_samples)

    # STANDARD APPROACH
    # outer_diversity, num_votes = outer_diversity_sampling(
    #     domain, num_samples=num_samples)

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
    sc_diversities = []

    optimal_diversities = []
    domain_sizes = []

    print("Computing diversities for different numbers of candidates...")

    for num_candidates in candidate_range:
        print(f"Processing {num_candidates} candidates...")
        sc_diversity, sc_size = compute_single_crossing_diversity(
            num_candidates, num_samples)
        sc_diversities.append(sc_diversity)
        domain_sizes.append(sc_size)

        if with_max:
            optimal_diversity = compute_optimal_diversity_for_size(
                num_candidates, sc_size, num_samples, max_iterations)
            optimal_diversities.append(optimal_diversity)

    # Export data to CSV for record-keeping

    results_dir = os.path.join(os.path.dirname(__file__), 'data', 'changing_m')
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, '_single_crossing.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['num_candidates',
                      'sc_diversity',
                      'gs_caterpillar_diversity',
                      'gs_balanced_diversity',
                      'optimal_diversity',
                       'domain_size']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i, num_candidates in enumerate(candidate_range):
            row = {
                'num_candidates': num_candidates,
                'sc_diversity': sc_diversities[i],
                'domain_size': domain_sizes[i],
                'optimal_diversity': optimal_diversities[i] if with_max else None
            }
            writer.writerow(row)

def plot_diversity_comparison(with_max=True):
    """Plot diversity comparison from CSV."""
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    results_dir = os.path.join(os.path.dirname(__file__), 'data', 'changing_m')
    csv_path = os.path.join(results_dir, '_single_crossing.csv')
    df = pd.read_csv(csv_path)
    candidate_range = df['num_candidates']
    sc_diversities = df['sc_diversity']
    optimal_diversities = df['optimal_diversity'] if with_max else None

    plt.figure(figsize=(10, 6))
    if with_max and optimal_diversities is not None:
        plt.plot(candidate_range, optimal_diversities,
                 label='~Maximum Possible (Simulated Annealing)',
                 marker='s',
                 linewidth=2,
                 markersize=8,
                 color='tab:green')

    plt.plot(candidate_range, sc_diversities,
             label='Single-Crossing Domain',
             marker='o',
             linewidth=2,
             markersize=8,
             color='tab:red')

    plt.xlabel('Number of Candidates', fontsize=22)
    plt.ylabel('Outer Diversity Score', fontsize=22)
    if with_max:
        plt.title('Outer Diversity: Single-Crossing vs Maximum Possible', fontsize=20)
    else:
        plt.title('Outer Diversity: Single-Crossing', fontsize=20)
    plt.legend(fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.xticks(candidate_range, fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('img/changing_m_single_crossing.png', dpi=300, bbox_inches='tight')
    plt.show()


def import_diversity_comparison_results():
    """Import diversity comparison results from CSV file."""
    import os
    import pandas as pd
    results_dir = os.path.join(os.path.dirname(__file__), 'data', 'changing_m')
    csv_path = os.path.join(results_dir, '_single_crossing.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Results file not found: {csv_path}")
    return pd.read_csv(csv_path)



if __name__ == "__main__":
    candidate_range = range(2, 20+1)
    num_samples = 1000
    max_iterations = 1
    from time import time
    start_time = time()
    compute_diversity_comparison_data(
        candidate_range, num_samples, max_iterations, with_max=False)
    end_time = time()
    print(f"Computation time: {end_time - start_time} seconds")
    plot_diversity_comparison(with_max=False)
