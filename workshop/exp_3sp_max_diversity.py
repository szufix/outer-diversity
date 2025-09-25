
from src.max_diversity.main import (
    compute_optimal_nodes,
    compute_optimal_nodes_single,
    merge_individual_results,
    check_individual_results_status,
)
from src.max_diversity.plot import plot_optimal_nodes_results


import sys
from time import time

def compute_single_domain_size():
    """
    Compute optimal nodes for a single domain size.
    Usage: python exp_3_optimal_votes_individual.py <num_candidates> <domain_size> <method_name>
    """
    if len(sys.argv) != 4:
        print("Usage: python exp_3_optimal_votes_individual.py <num_candidates> <domain_size> <method_name>")
        print("Example: python exp_3_optimal_votes_individual.py 5 10 ilp")
        sys.exit(1)

    num_candidates = int(sys.argv[1])
    domain_size = int(sys.argv[2])
    method_name = sys.argv[3]

    print(f"Computing {method_name} for {num_candidates} candidates, domain size {domain_size}")

    start = time()
    compute_optimal_nodes_single(num_candidates, domain_size, method_name)
    end = time()

    print(f"Computation completed in {end - start:.2f} seconds")

def merge_results():
    """
    Merge individual results into combined file.
    Usage: python exp_3_optimal_votes_individual.py merge <num_candidates> <method_name> <max_domain_size>
    """
    if len(sys.argv) != 5:
        print("Usage: python exp_3_optimal_votes_individual.py merge <num_candidates> <method_name> <max_domain_size>")
        print("Example: python exp_3_optimal_votes_individual.py merge 5 ilp 120")
        sys.exit(1)

    num_candidates = int(sys.argv[2])
    method_name = sys.argv[3]
    max_domain_size = int(sys.argv[4])

    domain_sizes = list(range(1, max_domain_size + 1))

    print(f"Checking status for {method_name}_m{num_candidates}...")
    available, missing = check_individual_results_status(num_candidates, method_name, domain_sizes)

    if available:
        print(f"Merging {len(available)} available results...")
        merge_individual_results(num_candidates, method_name, available)
        print("Merge completed!")
    else:
        print("No results available to merge.")

def check_status():
    """
    Check status of individual computations.
    Usage: python exp_3_optimal_votes_individual.py status <num_candidates> <method_name> <max_domain_size>
    """
    if len(sys.argv) != 5:
        print("Usage: python exp_3_optimal_votes_individual.py status <num_candidates> <method_name> <max_domain_size>")
        print("Example: python exp_3_optimal_votes_individual.py status 5 ilp 120")
        sys.exit(1)

    num_candidates = int(sys.argv[2])
    method_name = sys.argv[3]
    max_domain_size = int(sys.argv[4])

    domain_sizes = list(range(1, max_domain_size + 1))
    check_individual_results_status(num_candidates, method_name, domain_sizes)

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Available commands:")
        print("  python exp_3_optimal_votes_individual.py <num_candidates> <domain_size> <method_name>")
        print("  python exp_3_optimal_votes_individual.py merge <num_candidates> <method_name> <max_domain_size>")
        print("  python exp_3_optimal_votes_individual.py status <num_candidates> <method_name> <max_domain_size>")
        sys.exit(1)

    command = sys.argv[1]

    if command == "merge":
        merge_results()
    elif command == "status":
        check_status()
    else:
        # Assume it's num_candidates for single computation
        # sys.argv.insert(1, command)  # Put back the first argument
        compute_single_domain_size()

