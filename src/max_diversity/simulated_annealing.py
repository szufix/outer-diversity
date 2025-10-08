import networkx as nx
import numpy as np

from src.max_diversity.greedy import find_optimal_facilities_greedy
from src.diversity.sampling import outer_diversity_sampling
from src.diversity.sampling import spread_permutations
from src.max_diversity.utils import compute_total_cost, create_vote_integer_mapping


def find_optimal_facilities_sampled_simulated_annealing(
    m_candidates: int,
    m_facilities: int,
    max_iterations: int = 10000,
    initial_temp: float = 0.5,
    cooling_rate: float = 0.9,
    num_samples: int = 100,
    start_with: str = 'ic',
):
    """
    Simulated annealing using sampling instead of creating the full graph.
    Optimized to avoid generating all possible votes.
    """
    if m_facilities <= 0:
        return [], float('inf')

    # Initialize with random solution (vote tuples) - no need for all_votes
    current_facilities_votes = []
    if start_with == 'ic':
        for _ in range(m_facilities):
            candidates = list(range(m_candidates))
            random.shuffle(candidates)
            current_facilities_votes.append(tuple(candidates))
    elif start_with == 'grid':
        current_facilities_votes = spread_permutations(m_candidates, m_facilities)

    # Compute initial cost using sampling
    current_diversity, sampled_size = outer_diversity_sampling(current_facilities_votes, num_samples)

    best_facilities_votes = current_facilities_votes.copy()
    max_diversity = current_diversity

    temperature = initial_temp
    temp_update_freq = max(1, max_iterations // 1000)

    improvements = 0
    last_improvement = 0

    print(f"Sampled SA starting: cost={current_diversity}, facilities={len(current_facilities_votes)}")

    for iteration in range(max_iterations):
        # Generate neighbor solution by replacing one facility with a random vote
        new_facilities_votes = current_facilities_votes.copy()

        # Generate a new random vote instead of picking from all_votes
        facility_to_replace_idx = random.randint(0, len(current_facilities_votes) - 1)
        candidates = list(range(m_candidates))
        random.shuffle(candidates)
        new_vote = tuple(candidates)

        # Make sure the new vote is different from existing facilities
        max_attempts = 10
        attempts = 0
        while new_vote in current_facilities_votes and attempts < max_attempts:
            random.shuffle(candidates)
            new_vote = tuple(candidates)
            attempts += 1

        new_facilities_votes[facility_to_replace_idx] = new_vote

        # Compute cost using sampling
        new_cost, sampled_size = outer_diversity_sampling(new_facilities_votes, num_samples)

        # Accept or reject the new solution
        gain = new_cost - current_diversity

        if gain > 0:
            accept = True
            improvements += 1
            last_improvement = iteration
        elif temperature > 1e-10:
            try:
                prob = math.exp(gain / temperature)
                accept = random.random() < prob
            except (OverflowError, FloatingPointError):
                accept = False
        else:
            accept = False

        if accept:
            current_facilities_votes = new_facilities_votes
            current_diversity = new_cost

            # Update best solution if improved
            if current_diversity > max_diversity:
                best_facilities_votes = current_facilities_votes.copy()
                max_diversity = current_diversity
                print(f"Sampled SA improvement at iteration {iteration}: cost={max_diversity}")

        # Cool down temperature
        if iteration % temp_update_freq == 0:
            temperature *= cooling_rate

        # Early stopping conditions
        if temperature < 1e-10:
            print(f"Sampled SA stopped due to low temperature at iteration {iteration}")
            break

    print(f"Sampled SA final result for m={m_facilities}: cost={max_diversity}, improvements={improvements}, last_improvement at iteration {last_improvement}")


    return best_facilities_votes, max_diversity


def compute_total_cost_numpy(distance_matrix: np.ndarray, facilities_idx: np.ndarray) -> float:
    """
    Compute total cost efficiently using numpy operations.
    """
    if len(facilities_idx) == 0:
        return np.inf

    # Extract distances from all nodes to all facilities
    facility_distances = distance_matrix[:, facilities_idx]

    # Find minimum distance from each node to any facility
    min_distances = np.min(facility_distances, axis=1)

    return float(np.sum(min_distances))


import math
import random
import numpy as np
from typing import List, Tuple




def swap_distance(vote1: tuple, vote2: tuple) -> int:
    """
    Compute minimum number of adjacent swaps to transform vote1 into vote2.
    Uses bubble sort distance.
    """
    if vote1 == vote2:
        return 0

    # Convert to lists for manipulation
    current = list(vote1)
    target = list(vote2)

    swaps = 0
    n = len(current)

    # Use bubble sort to count swaps
    for i in range(n):
        for j in range(n - 1 - i):
            if current[j] != target[j]:
                # Find where target[j] is in current
                target_pos = current.index(target[j])
                # Bubble it to position j
                while target_pos > j:
                    current[target_pos], current[target_pos - 1] = current[target_pos - 1], current[target_pos]
                    target_pos -= 1
                    swaps += 1
                break

    return swaps


def find_optimal_facilities_simulated_annealing(graph: nx.Graph, m: int,
                                               max_iterations: int = 10000,
                                               initial_temp: float = 0.5,
                                               cooling_rate: float = 0.9,
                                                num_candidates=None) -> Tuple[List[int], int]:
    """
    Find facilities using simulated annealing to approximate optimal solution.
    Fixed to use consistent cost calculation.
    """
    nodes = list(graph.nodes())
    n = len(nodes)

    if m >= n:
        return nodes, 0
    if m <= 0:
        return [], compute_total_cost(graph, set())

    # Start with greedy solution instead of random
    try:
        greedy_facilities, greedy_cost = find_optimal_facilities_greedy(graph, m)
        current_facilities = greedy_facilities
        print(f"SA starting with greedy solution: cost={greedy_cost}")
    except:
        # Fallback to random if greedy fails
        current_facilities = random.sample(nodes, m)

    # Use consistent graph-based cost calculation throughout
    current_cost = compute_total_cost(graph, set(current_facilities))
    best_facilities = current_facilities.copy()
    best_cost = current_cost

    n = len(graph.nodes)
    best_cost = 1 - best_cost / n * 2 / math.comb(num_candidates, 2)

    temperature = initial_temp
    temp_update_freq = max(1, max_iterations // 10000)

    improvements = 0
    stagnation_counter = 0
    last_improvement = 0

    for iteration in range(max_iterations):
        # Generate neighbor solution by replacing one facility
        new_facilities = current_facilities.copy()

        # Get available nodes (not currently facilities)
        available_nodes = [node for node in nodes if node not in current_facilities]

        if len(available_nodes) > 0:
            # Randomly select facility to remove and node to add
            facility_to_remove_idx = random.randint(0, len(current_facilities) - 1)
            node_to_add = random.choice(available_nodes)

            new_facilities[facility_to_remove_idx] = node_to_add

            # Use consistent graph-based cost calculation
            new_cost = compute_total_cost(graph, set(new_facilities))
            n = len(graph.nodes)
            new_cost = 1 - new_cost / n * 2 / math.comb(num_candidates, 2)


            # Accept or reject the new solution
            gain = new_cost - current_cost

            if gain > 0:
                accept = True
                improvements += 1
                stagnation_counter = 0
                last_improvement = iteration
            elif temperature > 1e-10:
                try:
                    prob = math.exp(gain / temperature)
                    print(prob, temperature)
                    accept = random.random() < prob
                except (OverflowError, FloatingPointError):
                    accept = False
            else:
                accept = False

            if accept:
                current_facilities = new_facilities
                current_cost = new_cost

                # Update best solution if improved
                if current_cost > best_cost:
                    best_facilities = current_facilities.copy()
                    best_cost = current_cost
                    print(f"SA improvement at iteration {iteration}: cost={best_cost}")
            else:
                stagnation_counter += 1

        # Cool down temperature
        if iteration % temp_update_freq == 0:
            temperature *= cooling_rate

        # Early stopping conditions
        if temperature < 1e-10:
            print(f"SA stopped due to low temperature at iteration {iteration}")
            break

        if stagnation_counter > max_iterations // 10:
            print(f"SA stopped due to stagnation at iteration {iteration}")
            break

    # Verify the final cost (should match best_cost)
    final_cost = compute_total_cost(graph, set(best_facilities))

    if final_cost != best_cost:
        print(f"WARNING: SA cost mismatch! best_cost={best_cost}, final_cost={final_cost}")
        # Use the verified cost
        best_cost = final_cost

    print(f"SA final result for m={m}: cost={best_cost}, improvements={improvements}, last_improvement at iteration {last_improvement}")

    return best_facilities, best_cost


