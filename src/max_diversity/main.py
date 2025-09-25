import csv
import os
import math
from typing import List, Dict

from src.max_diversity.bruteforce import find_optimal_facilities_bruteforce
from src.max_diversity.ilp import find_optimal_ilp, find_optimal_facilities_milp_approx, \
    find_optimal_facilities_greedy_ilp, find_optimal_facilities_greedy_ilp_fast
from src.max_diversity.simulated_annealing import find_optimal_facilities_simulated_annealing, \
    find_optimal_facilities_sampled_simulated_annealing
from src.max_diversity.swap_graph import create_vote_swap_graph
from src.max_diversity.impartial_culture import diversity_for_ic, diversity_for_smpl_ic


def compute_optimal_nodes(
        num_candidates,
        domain_sizes,
        method_name,
        max_iterations=None,
        num_samples=None,
        start_with='ic'):
    import csv
    import os
    results = []

    # Prepare output file at the beginning
    os.makedirs('data/optimal_nodes', exist_ok=True)
    csv_filename = f'data/optimal_nodes/{method_name}_m{num_candidates}.csv'
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['domain_size', 'total_cost', 'optimal_nodes_int', 'optimal_nodes_votes']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    vote_graph = None
    vote_to_int = None
    int_to_vote = None
    # Always define these so they're not referenced before assignment
    if method_name not in ['smpl_sa']:
        print("create a graph")
        vote_graph, vote_to_int, int_to_vote = create_vote_swap_graph(num_candidates)
        print("graph created")



    previous_nodes = []

    for domain_size in domain_sizes:
        print(f"Processing domain size: {domain_size}")
        optimal_nodes_votes = []  # Always define before use
        if method_name == 'ilp':
            optimal_nodes, total_cost = find_optimal_ilp(
                                                vote_graph, domain_size)
        elif method_name == 'ilp_fast':
            optimal_nodes, total_cost = find_optimal_facilities_milp_approx(
                                                vote_graph, domain_size)
        elif method_name == 'greedy_ilp':
            optimal_nodes, total_cost = find_optimal_facilities_greedy_ilp(
                                                vote_graph, domain_size, previous_nodes)
        elif method_name == 'greedy_ilp_fast':
            optimal_nodes, total_cost = find_optimal_facilities_greedy_ilp_fast(
                                                vote_graph, domain_size, previous_nodes)
        elif method_name == 'sa':
            optimal_nodes, total_cost = find_optimal_facilities_simulated_annealing(
                vote_graph, domain_size, max_iterations=max_iterations)
        elif method_name == 'smpl_sa':
            optimal_nodes_votes, total_cost = find_optimal_facilities_sampled_simulated_annealing(
                num_candidates, domain_size, max_iterations=max_iterations, num_samples=num_samples,
                start_with=start_with)
            optimal_nodes = []

        elif method_name == 'ic':
            optimal_nodes, total_cost = diversity_for_ic(
                vote_graph, domain_size)

        elif method_name == 'smpl_ic':
            optimal_nodes_votes, total_cost = diversity_for_smpl_ic(
                num_candidates, domain_size, num_samples=num_samples)
            optimal_nodes = []


        elif method_name == 'bf':
            optimal_nodes, total_cost = find_optimal_facilities_bruteforce(
                                                vote_graph, domain_size)
        else:
            raise ValueError(f"Unknown method: {method_name}")

        # Store results
        if method_name in ['smpl_sa', 'smpl_ic']:
            result = {
                'domain_size': domain_size,
                'total_cost': total_cost,
                'optimal_nodes_int': str([]),
                'optimal_nodes_votes': str(optimal_nodes_votes),
            }
            previous_nodes = []  # Reset for sampled SA since we don't use graph nodes
        else:
            n = len(vote_graph.nodes)
            total_cost = 1 - total_cost / n * 2 / math.comb(num_candidates, 2)

            result = {
                'domain_size': domain_size,
                'total_cost': total_cost,
                'optimal_nodes_int': str(optimal_nodes),
                'optimal_nodes_votes': str([int_to_vote[f] for f in optimal_nodes]) if int_to_vote is not None else str([]),
            }
            previous_nodes = optimal_nodes

        results.append(result)
        print(f"  Result: {method_name} found cost={total_cost} for domain_size={domain_size}")

        # Append result to file immediately
        with open(csv_filename, 'a', newline='') as csvfile:
            fieldnames = ['domain_size', 'total_cost', 'optimal_nodes_int', 'optimal_nodes_votes']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(result)

    # Save all results to file (optional, for compatibility)
    # save_optimal_nodes_results(results, num_candidates, method_name)


def compute_optimal_nodes_single(num_candidates: int, domain_size: int, method_name: str):
    """
    Compute optimal nodes for a single domain size and save to individual file.

    Args:
        num_candidates: Number of candidates
        domain_size: Single domain size to compute
        method_name: Method name ('ilp', 'sa', etc.)
    """
    print(f"Computing {method_name} for m={num_candidates}, domain_size={domain_size}")

    vote_graph = None
    vote_to_int = None
    int_to_vote = None
    if method_name not in ['smpl_sa']:
        print("Creating graph...")
        vote_graph, vote_to_int, int_to_vote = create_vote_swap_graph(num_candidates)
        print("Graph created")

    optimal_nodes_votes = []
    if method_name == 'ilp':
        optimal_nodes, total_cost = find_optimal_ilp(vote_graph, domain_size)
    elif method_name == 'ilp_fast':
        optimal_nodes, total_cost = find_optimal_facilities_milp_approx(vote_graph, domain_size)
    # elif method_name == 'sa':
    #     optimal_nodes, total_cost = find_optimal_facilities_simulated_annealing(
    #         vote_graph, domain_size, max_iterations=1000)
    # elif method_name == 'smpl_sa':
    #     optimal_nodes_votes, total_cost = find_optimal_facilities_sampled_simulated_annealing(
    #         num_candidates, domain_size, max_iterations=1000, num_samples=1000)
    #     optimal_nodes = []
    else:
        raise ValueError(f"Unknown method: {method_name}")

    # Create result
    if method_name == 'smpl_sa':
        result = {
            'domain_size': domain_size,
            'total_cost': total_cost,
            'optimal_nodes_int': str([]),
            'optimal_nodes_votes': str(optimal_nodes_votes),
        }
    else:
        result = {
            'domain_size': domain_size,
            'total_cost': total_cost,
            'optimal_nodes_int': str(optimal_nodes),
            'optimal_nodes_votes': str([int_to_vote[f] for f in optimal_nodes]) if int_to_vote is not None else str([]),
        }

    # Save to individual file
    save_single_result(result, num_candidates, method_name, domain_size)
    print(f"Saved result: {method_name} cost={total_cost} for domain_size={domain_size}")


def save_single_result(result: Dict, num_candidates: int, method_name: str, domain_size: int):
    """
    Save a single result to an individual CSV file.

    Args:
        result: Result dictionary
        num_candidates: Number of candidates
        method_name: Method name
        domain_size: Domain size for this result
    """
    # Create directory if it doesn't exist
    individual_dir = f'data/optimal_nodes/individual/{method_name}_m{num_candidates}'
    os.makedirs(individual_dir, exist_ok=True)

    csv_filename = f'{individual_dir}/domain_size_{domain_size}.csv'

    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['domain_size', 'total_cost', 'optimal_nodes_int', 'optimal_nodes_votes']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(result)

    print(f"Saved individual result to {csv_filename}")


def load_single_result(num_candidates: int, method_name: str, domain_size: int) -> Dict:
    """
    Load a single result from individual file.

    Args:
        num_candidates: Number of candidates
        method_name: Method name
        domain_size: Domain size to load

    Returns:
        Result dictionary or None if file doesn't exist
    """
    individual_dir = f'data/optimal_nodes/individual/{method_name}_m{num_candidates}'
    csv_filename = f'{individual_dir}/domain_size_{domain_size}.csv'

    try:
        with open(csv_filename, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            result = next(reader)
            return {
                'domain_size': int(result['domain_size']),
                'total_cost': int(result['total_cost']),
                'optimal_nodes_int': eval(result['optimal_nodes_int']),
                'optimal_nodes_votes': eval(result['optimal_nodes_votes'])
            }
    except (FileNotFoundError, StopIteration):
        return {}


def merge_individual_results(num_candidates: int, method_name: str, domain_sizes: List[int]):
    """
    Merge individual result files into a single combined file.

    Args:
        num_candidates: Number of candidates
        method_name: Method name
        domain_sizes: List of domain sizes to merge
    """
    results = []
    missing_sizes = []

    for domain_size in domain_sizes:
        result = load_single_result(num_candidates, method_name, domain_size)
        if result:
            results.append(result)
        else:
            missing_sizes.append(domain_size)

    if missing_sizes:
        print(f"Warning: Missing results for domain sizes: {missing_sizes}")

    if results:
        # Sort by domain size
        results.sort(key=lambda x: x['domain_size'])

        # Save merged results
        save_optimal_nodes_results(results, num_candidates, method_name)
        print(f"Merged {len(results)} individual results into combined file for {method_name}_m{num_candidates}")
    else:
        print(f"No individual results found to merge for {method_name}_m{num_candidates}")


def check_individual_results_status(num_candidates: int, method_name: str, domain_sizes: List[int]):
    """
    Check which individual results are available and which are missing.

    Args:
        num_candidates: Number of candidates
        method_name: Method name
        domain_sizes: List of domain sizes to check
    """
    available = []
    missing = []

    for domain_size in domain_sizes:
        result = load_single_result(num_candidates, method_name, domain_size)
        if result:
            available.append(domain_size)
        else:
            missing.append(domain_size)

    print(f"Status for {method_name}_m{num_candidates}:")
    print(f"  Available: {available}")
    print(f"  Missing: {missing}")
    print(f"  Progress: {len(available)}/{len(domain_sizes)} ({100*len(available)/len(domain_sizes):.1f}%)")

    return available, missing


def save_optimal_nodes_results(results: List[Dict], num_candidates: int, method_name: str):
    """
    Save optimal nodes results to CSV file.

    Args:
        results: List of result dictionaries
        num_candidates: Number of candidates used
        method_name: Method name for filename
    """
    # Create directory if it doesn't exist
    os.makedirs('data/optimal_nodes', exist_ok=True)

    csv_filename = f'data/optimal_nodes/{method_name}_m{num_candidates}.csv'

    with open(csv_filename, 'w', newline='') as csvfile:
        if results:
            fieldnames = ['domain_size', 'total_cost', 'optimal_nodes_int', 'optimal_nodes_votes']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for result in results:
                writer.writerow(result)

    print(f"Saved {len(results)} results to {csv_filename}")
