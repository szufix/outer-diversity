from collections import deque
import networkx as nx
from typing import List, Set, Tuple, Dict
import gurobipy as gp
from gurobipy import GRB
from itertools import permutations
import csv
import os
import matplotlib.pyplot as plt
from src.print_utils import COLOR,LABEL
import random
import math
import numpy as np

from src.outer_diversity import normalization


import networkx as nx
import gurobipy as gp
from gurobipy import GRB
from typing import List, Tuple

def vote_to_integer(vote: tuple, m: int) -> int:
    """
    Convert a vote (permutation) to a unique integer.
    Uses factorial number system (Lehmer code).

    Args:
        vote: Tuple representing a permutation
        m: Number of candidates

    Returns:
        Unique integer representing the vote
    """
    candidates = list(range(m))
    result = 0
    factorial = 1

    for i in range(m-1, 0, -1):
        pos = candidates.index(vote[i])
        result += pos * factorial
        candidates.pop(pos)
        factorial *= (m - i + 1)

    return result

def integer_to_vote(code: int, m: int) -> tuple:
    """
    Convert an integer back to a vote (permutation).

    Args:
        code: Integer representing the vote
        m: Number of candidates

    Returns:
        Tuple representing the permutation
    """
    candidates = list(range(m))
    result = []

    for i in range(m-1, 0, -1):
        factorial = 1
        for j in range(1, i+1):
            factorial *= j

        pos = code // factorial
        result.append(candidates.pop(pos))
        code %= factorial

    result.append(candidates[0])
    return tuple(reversed(result))

def create_vote_integer_mapping(m: int) -> Tuple[Dict[tuple, int], Dict[int, tuple]]:
    """
    Create bidirectional mapping between votes and integers.

    Args:
        m: Number of candidates

    Returns:
        Tuple of (vote_to_int_map, int_to_vote_map)
    """
    candidates = list(range(m))
    all_votes = list(permutations(candidates))

    vote_to_int = {}
    int_to_vote = {}

    for i, vote in enumerate(all_votes):
        vote_to_int[vote] = i
        int_to_vote[i] = vote

    return vote_to_int, int_to_vote

def bfs_distances(graph: nx.Graph, start: int) -> dict:
    """Compute shortest distances from start node to all other nodes using BFS."""
    distances = {}
    queue = deque([(start, 0)])
    visited = {start}

    while queue:
        node, dist = queue.popleft()
        distances[node] = dist

        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))

    return distances

def compute_total_cost(graph: nx.Graph, facilities: Set[int]) -> int:
    """Compute total cost: sum of distances from each node to closest facility."""
    if not facilities:
        return float('inf')

    total_cost = 0
    for node in graph.nodes():
        min_dist = float('inf')
        for facility in facilities:
            # Use NetworkX shortest path for simplicity
            dist = nx.shortest_path_length(graph, node, facility)
            min_dist = min(min_dist, dist)
        total_cost += min_dist

    return total_cost

def find_optimal_facilities_greedy(graph: nx.Graph, m: int) -> Tuple[List[int], int]:
    """
    Find m facilities that minimize total distance cost using greedy algorithm.

    Returns:
        Tuple of (facility_nodes, total_cost)
    """
    if m <= 0:
        return [], float('inf')

    facilities = set()
    nodes = list(graph.nodes())

    # Precompute all pairwise distances
    all_distances = {}
    for node in nodes:
        all_distances[node] = bfs_distances(graph, node)

    # Greedy selection
    for _ in range(m):
        best_node = None
        best_cost = float('inf')

        for candidate in nodes:
            if candidate in facilities:
                continue

            # Try adding this candidate
            temp_facilities = facilities | {candidate}

            # Compute cost with this facility set
            cost = 0
            for node in nodes:
                min_dist = min(all_distances[f][node] for f in temp_facilities)
                cost += min_dist

            if cost < best_cost:
                best_cost = cost
                best_node = candidate

        if best_node is not None:
            facilities.add(best_node)

    final_cost = compute_total_cost(graph, facilities)
    return list(facilities), final_cost

def find_optimal_facilities_bruteforce(graph: nx.Graph, m: int) -> Tuple[List[int], int]:
    """
    Find optimal facilities using brute force (for small graphs only).
    """
    from itertools import combinations

    nodes = list(graph.nodes())
    if m > len(nodes):
        return nodes, compute_total_cost(graph, set(nodes))

    best_facilities = []
    best_cost = float('inf')

    for facility_combo in combinations(nodes, m):
        cost = compute_total_cost(graph, set(facility_combo))
        if cost < best_cost:
            best_cost = cost
            best_facilities = list(facility_combo)

    return best_facilities, best_cost

def find_optimal_facilities_milp_approx(graph: nx.Graph, m: int) -> Tuple[List[int], int]:
    """
    Find optimal facilities using MILP with Gurobi (optimized version).

    Returns:
        Tuple of (facility_nodes, total_cost)
    """
    nodes = list(graph.nodes())
    n = len(nodes)

    # Early termination for trivial cases
    if m >= n:
        return nodes, 0
    if m <= 0:
        return [], sum(nx.eccentricity(graph).values())

    # Precompute all pairwise distances
    distances = {}
    for i in nodes:
        distances[i] = nx.single_source_shortest_path_length(graph, i)

    # Create model with optimized settings
    model = gp.Model("facility_location")
    model.setParam('OutputFlag', 0)    # Suppress output
    model.setParam('Threads', 0)      # Use all available threads
    model.setParam('MIPGap', 0.01)     # Allow 1% optimality gap for speed
    model.setParam('TimeLimit', 300)   # 5 minute time limit
    model.setParam('Cuts', 2)          # Aggressive cuts
    model.setParam('Heuristics', 0.25)  # Spend 20% time on heuristics
    model.setParam('MIPFocus', 1)      # Focus on finding feasible solutions quickly

    # Decision variables
    x = model.addVars(nodes, vtype=GRB.BINARY, name="facility")

    # Use continuous variables for assignment (will be automatically integral at optimum)
    y = model.addVars(nodes, nodes, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="assignment")

    # Objective: minimize total distance
    obj = gp.quicksum(distances[j][i] * y[i, j] for i in nodes for j in nodes)
    model.setObjective(obj, GRB.MINIMIZE)

    # Constraints
    # Each node must be assigned to exactly one facility
    for i in nodes:
        model.addConstr(gp.quicksum(y[i, j] for j in nodes) == 1, f"assignment_{i}")

    # Can only assign to open facilities
    for i in nodes:
        for j in nodes:
            model.addConstr(y[i, j] <= x[j], f"facility_open_{i}_{j}")

    # Exactly m facilities must be opened
    model.addConstr(gp.quicksum(x[j] for j in nodes) == m, "facility_count")

    # Add valid inequalities to tighten the formulation
    # Lower bound: each node must be served by some facility
    total_min_distance = sum(min(distances[j][i] for j in nodes) for i in nodes)
    model.addConstr(obj >= total_min_distance, "lower_bound")

    # Symmetry breaking: prefer lower-indexed nodes when costs are equal
    if len(nodes) > 1 and m > 1:
        sorted_nodes = sorted(nodes)
        for i in range(min(len(sorted_nodes)-1, m-1)):
            model.addConstr(x[sorted_nodes[i]] >= x[sorted_nodes[i+1]], f"symmetry_{i}")

    # Warm start with greedy solution
    try:
        greedy_facilities, greedy_cost = find_optimal_facilities_greedy(graph, m)

        # Set facility variables
        for node in nodes:
            if node in greedy_facilities:
                x[node].start = 1
            else:
                x[node].start = 0

        # Set assignment variables
        for i in nodes:
            closest_facility = min(greedy_facilities,
                                 key=lambda f: distances[f][i])
            for j in nodes:
                if j == closest_facility:
                    y[i, j].start = 1
                else:
                    y[i, j].start = 0
    except:
        print("error in warm start")
        pass  # If greedy fails, continue without warm start

    # Solve
    model.optimize()

    if model.status == GRB.OPTIMAL:
        # Select top m facilities with highest x values
        facility_values = [(j, x[j].x) for j in nodes]
        facility_values.sort(key=lambda item: item[1], reverse=True)
        facilities = [facility_values[i][0] for i in range(m)]
        total_cost = int(model.objVal)
        return facilities, total_cost
    elif model.status == GRB.TIME_LIMIT:
        # Return best solution found so far
        if model.solCount > 0:
            # Select top m facilities with highest x values
            facility_values = [(j, x[j].x) for j in nodes]
            facility_values.sort(key=lambda item: item[1], reverse=True)
            facilities = [facility_values[i][0] for i in range(m)]
            total_cost = int(model.objVal)
            print(f"Time limit reached for m={m}. Best solution found: {total_cost} (gap: {model.MIPGap:.2%})")
            return facilities, total_cost

    print(f"MILP failed with status: {model.status}")
    return [], float('inf')


def find_optimal_ilp(graph: nx.Graph, m: int) -> Tuple[List[int], int]:
    """
    Solve the facility location problem via ILP:
    - Select exactly m facilities.
    - Minimize total distance from each node to its assigned facility.

    Args:
        graph: NetworkX graph (unweighted, undirected).
        m: number of facilities to open.

    Returns:
        (facilities, total_cost)
        facilities: list of chosen facility node indices.
        total_cost: objective value (sum of assignment distances).
    """
    nodes = list(graph.nodes())
    n = len(nodes)

    # Precompute all-pairs shortest path distances
    distances = {i: nx.single_source_shortest_path_length(graph, i) for i in nodes}
    max_distance = max(max(d.values()) for d in distances.values())

    # Create model
    model = gp.Model("compact_facility_location")
    model.setParam("OutputFlag", 0)

    # Decision variables
    x = model.addVars(nodes, vtype=GRB.BINARY, name="facility")        # facility open
    y = model.addVars(nodes, nodes, vtype=GRB.BINARY, name="assign")   # assignment
    d = model.addVars(nodes, vtype=GRB.INTEGER, lb=0, ub=max_distance, name="dist")

    # Objective: minimize total distance
    model.setObjective(gp.quicksum(d[i] for i in nodes), GRB.MINIMIZE)

    # Constraints
    model.addConstrs((gp.quicksum(y[i, j] for j in nodes) == 1 for i in nodes), name="serve")
    model.addConstrs((y[i, j] <= x[j] for i in nodes for j in nodes), name="open")
    model.addConstrs(
        (d[i] == gp.quicksum(distances[j][i] * y[i, j] for j in nodes) for i in nodes),
        name="dist_consistency"
    )
    model.addConstr(gp.quicksum(x[j] for j in nodes) == m, name="facility_count")

    # Warm start with greedy heuristic
    try:
        greedy_facilities, _ = find_optimal_facilities_greedy(graph, m)
        for j in nodes:
            x[j].start = int(j in greedy_facilities)
        for i in nodes:
            closest = min(greedy_facilities, key=lambda f: distances[f][i])
            for j in nodes:
                y[i, j].start = int(j == closest)
            d[i].start = distances[closest][i]
    except Exception as e:
        print(f"Warm start failed: {e}")

    # Solve
    model.optimize()

    def extract_solution() -> Tuple[List[int], int]:
        facilities = [j for j in nodes if x[j].X > 0.5]
        return facilities, int(model.ObjVal)

    if model.status == GRB.OPTIMAL:
        return extract_solution()
    if model.status == GRB.TIME_LIMIT and model.SolCount > 0:
        facilities, cost = extract_solution()
        print(f"MILP time limit reached for m={m}. Best: {cost}")
        return facilities, cost

    print(f"MILP failed with status {model.status} for m={m}")
    try:
        return find_optimal_facilities_greedy(graph, m)
    except Exception:
        return [], float("inf")


def create_vote_swap_graph(m: int) -> Tuple[nx.Graph, Dict[tuple, int], Dict[int, tuple]]:
    """
    Create a graph where nodes are integers representing votes (permutations of m candidates)
    and edges connect votes that differ by exactly one adjacent swap.

    Args:
        m: Number of candidates

    Returns:
        Tuple of (NetworkX graph with integer nodes, vote_to_int_map, int_to_vote_map)
    """
    # Generate all possible votes (permutations)
    candidates = list(range(m))
    all_votes = list(permutations(candidates))

    # Create mappings
    vote_to_int, int_to_vote = create_vote_integer_mapping(m)

    # Create graph with integer nodes
    G = nx.Graph()

    # Add all votes as integer nodes
    for vote in all_votes:
        G.add_node(vote_to_int[vote])

    # Add edges between votes that differ by one adjacent swap
    for i, vote1 in enumerate(all_votes):
        # print(i)
        for vote2 in all_votes[i+1:]:
            if is_one_swap_distance(vote1, vote2):
                G.add_edge(vote_to_int[vote1], vote_to_int[vote2])

    return G, vote_to_int, int_to_vote

def is_one_swap_distance(vote1: tuple, vote2: tuple) -> bool:
    """
    Check if two votes differ by exactly one adjacent swap.

    Args:
        vote1, vote2: Tuples representing votes (permutations)

    Returns:
        True if votes differ by exactly one adjacent swap
    """
    if len(vote1) != len(vote2):
        return False

    differences = []
    for i in range(len(vote1)):
        if vote1[i] != vote2[i]:
            differences.append(i)

    # Must have exactly 2 differences
    if len(differences) != 2:
        return False

    # Differences must be adjacent positions
    pos1, pos2 = differences
    if abs(pos1 - pos2) != 1:
        return False

    # Elements at different positions must be swapped
    return vote1[pos1] == vote2[pos2] and vote1[pos2] == vote2[pos1]

def save_optimal_nodes_results(
        results: List[Dict],
        num_candidates: int,
        method_name):
    """
    Save optimal nodes results to CSV file.

    Args:
        results: List of dictionaries containing results for each domain size
        num_candidates: Number of candidates used
    """
    # Create data directory if it doesn't exist
    os.makedirs('data/optimal_nodes', exist_ok=True)

    csv_filename = f'data/optimal_nodes/{method_name}_m{num_candidates}.csv'

    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['domain_size',
                      'total_cost',
                      'optimal_nodes_int',
                     'optimal_nodes_votes']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(result)

    print(f"Results saved to {csv_filename}")

def find_optimal_facilities_simulated_annealing(graph: nx.Graph, m: int,
                                               max_iterations: int = 10000,
                                               initial_temp: float = 100.0,
                                               cooling_rate: float = 0.9) -> Tuple[List[int], int]:
    """
    Find facilities using simulated annealing to approximate optimal solution.
    Optimized version using numpy for faster computations.

    Args:
        graph: NetworkX graph
        m: Number of facilities to place
        max_iterations: Maximum number of iterations
        initial_temp: Initial temperature for annealing
        cooling_rate: Rate at which temperature decreases

    Returns:
        Tuple of (facility_nodes, total_cost)
    """
    nodes = list(graph.nodes())
    n = len(nodes)

    if m > n:
        return nodes, compute_total_cost(graph, set(nodes))
    # Precompute all pairwise distances as numpy array for faster access
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    distance_matrix = np.full((n, n), np.inf)

    for node in nodes:
        distances = nx.single_source_shortest_path_length(graph, node)
        node_idx = node_to_idx[node]
        for target, dist in distances.items():
            target_idx = node_to_idx[target]
            distance_matrix[node_idx, target_idx] = dist
    # Initialize with random solution
    current_facilities_idx = np.random.choice(n, size=m, replace=False)
    current_cost = compute_total_cost_numpy(distance_matrix, current_facilities_idx)
    best_facilities_idx = current_facilities_idx.copy()
    best_cost = current_cost

    temperature = initial_temp

    # Pre-allocate arrays for efficiency
    available_nodes = np.arange(n)

    # Cooling schedule - update temperature less frequently for efficiency
    temp_update_freq = max(1, max_iterations // 1000)

    for iteration in range(max_iterations):
        # print(iteration)
        # Generate neighbor solution by replacing one facility
        new_facilities_idx = current_facilities_idx.copy()

        # Get available nodes (not currently facilities)
        facility_mask = np.zeros(n, dtype=bool)
        facility_mask[current_facilities_idx] = True
        available_idx = available_nodes[~facility_mask]

        if len(available_idx) > 0:
            # Randomly select facility to remove and node to add
            facility_to_remove_pos = np.random.randint(m)
            node_to_add_idx = np.random.choice(available_idx)

            new_facilities_idx[facility_to_remove_pos] = node_to_add_idx

            new_cost = compute_total_cost_numpy(distance_matrix, new_facilities_idx)

            # Accept or reject the new solution
            cost_diff = new_cost - current_cost

            if cost_diff < 0 or np.random.random() < np.exp(-cost_diff / temperature):
                current_facilities_idx = new_facilities_idx
                current_cost = new_cost

                # Update best solution if improved
                if current_cost < best_cost:
                    best_facilities_idx = current_facilities_idx.copy()
                    best_cost = current_cost

        # Cool down temperature (less frequently for efficiency)
        if iteration % temp_update_freq == 0:
            temperature *= cooling_rate

            # Early stopping if temperature is very low
            if temperature < 0.01:
                print(f"Early stopping at iteration {iteration}")
                break

    # Convert back to original node indices
    best_facilities = [nodes[idx] for idx in best_facilities_idx]
    return best_facilities, int(best_cost)

def compute_total_cost_numpy(distance_matrix: np.ndarray, facilities_idx: np.ndarray) -> float:
    """
    Compute total cost efficiently using numpy operations.

    Args:
        distance_matrix: n x n matrix of distances between all node pairs
        facilities_idx: Array of facility indices

    Returns:
        Total cost (sum of distances from each node to closest facility)
    """
    if len(facilities_idx) == 0:
        return np.inf

    # Extract distances from all nodes to all facilities
    facility_distances = distance_matrix[:, facilities_idx]

    # Find minimum distance from each node to any facility
    min_distances = np.min(facility_distances, axis=1)

    return np.sum(min_distances)

def find_optimal_facilities_greedy_ilp(
        graph: nx.Graph, m: int, previous_nodes) -> Tuple[List[int], int]:
    """
    Find m facilities using greedy ILP approach - at each iteration find one new facility
    that minimizes the total cost when added to the current facility set.

    Args:
        graph: NetworkX graph
        m: Number of facilities to place

    Returns:
        Tuple of (facility_nodes, total_cost)
    """
    nodes = list(graph.nodes())
    n = len(nodes)

    # Precompute all pairwise distances
    distances = {}
    for i in nodes:
        distances[i] = nx.single_source_shortest_path_length(graph, i)

    facilities = previous_nodes


    # Find the best facility to add using ILP
    best_facility = None
    best_cost = float('inf')

    # Try each remaining node as a potential facility
    remaining_nodes = [node for node in nodes if node not in facilities]

    for candidate in remaining_nodes:
        # Solve ILP with current facilities + candidate
        current_facilities = facilities + [candidate]

        # Create model for assignment problem with fixed facilities
        model = gp.Model(f"greedy_ilp_iter_{m}")
        model.setParam('OutputFlag', 0)
        # model.setParam('Threads', -1)
        # model.setParam('TimeLimit', 60)  # 1 minute per iteration

        # Assignment variables: y[i,j] = 1 if node i is assigned to facility j
        y = model.addVars(nodes, current_facilities, vtype=GRB.BINARY, name="assignment")

        # Objective: minimize total assignment cost
        obj = gp.quicksum(distances[j][i] * y[i, j] for i in nodes for j in current_facilities)
        model.setObjective(obj, GRB.MINIMIZE)

        # Each node must be assigned to exactly one facility
        for i in nodes:
            model.addConstr(gp.quicksum(y[i, j] for j in current_facilities) == 1, f"assign_{i}")

        # Solve the assignment problem
        model.optimize()

        if model.status == GRB.OPTIMAL:
            cost = int(model.objVal)
            if cost < best_cost:
                best_cost = cost
                best_facility = candidate
        else:
            print(f"Assignment problem failed for candidate {candidate}")

    # Add the best facility found
    if best_facility is not None:
        facilities.append(best_facility)
        print(f"Added facility {best_facility}, current cost: {best_cost}")
    else:
        print(f"No valid facility found at iteration {m + 1}")

    # Compute final cost
    final_cost = compute_total_cost(graph, set(facilities))
    return facilities, final_cost

def find_optimal_facilities_greedy_ilp_fast(
        graph: nx.Graph, m: int, previos_nodes) -> Tuple[List[int], int]:
    """
    Faster version of greedy ILP that uses continuous relaxation for speed.

    Args:
        graph: NetworkX graph
        m: Number of facilities to place

    Returns:
        Tuple of (facility_nodes, total_cost)
    """
    nodes = list(graph.nodes())
    n = len(nodes)

    # Precompute all pairwise distances
    distances = {}
    for i in nodes:
        distances[i] = nx.single_source_shortest_path_length(graph, i)

    facilities = previos_nodes


    best_facility = None
    best_cost = float('inf')

    remaining_nodes = [node for node in nodes if node not in facilities]

    for candidate in remaining_nodes:
        current_facilities = facilities + [candidate]

        # Create LP relaxation model for faster solving
        model = gp.Model(f"greedy_ilp_fast_iter_{m}")
        model.setParam('OutputFlag', 0)
        model.setParam('Threads', 0)
        model.setParam('TimeLimit', 30)  # 30 seconds per iteration

        # Continuous assignment variables (LP relaxation)
        y = model.addVars(nodes, current_facilities, vtype=GRB.CONTINUOUS,
                         lb=0, ub=1, name="assignment")

        # Objective: minimize total assignment cost
        obj = gp.quicksum(distances[j][i] * y[i, j] for i in nodes for j in current_facilities)
        model.setObjective(obj, GRB.MINIMIZE)

        # Each node must be assigned to exactly one facility
        for i in nodes:
            model.addConstr(gp.quicksum(y[i, j] for j in current_facilities) == 1, f"assign_{i}")

        # Solve the LP relaxation
        model.optimize()

        if model.status == GRB.OPTIMAL:
            cost = model.objVal  # LP relaxation gives lower bound
            if cost < best_cost:
                best_cost = cost
                best_facility = candidate
        else:
            print(f"LP relaxation failed for candidate {candidate}")

    # Add the best facility found
    if best_facility is not None:
        facilities.append(best_facility)
        print(f"Added facility {best_facility}, LP lower bound: {best_cost:.2f}")
    else:
        print(f"No valid facility found at iteration {m + 1}")

    # Compute final cost using actual integer assignment
    final_cost = compute_total_cost(graph, set(facilities))
    return facilities, final_cost

import math
import random
import numpy as np
from typing import List, Tuple

def find_optimal_facilities_simulated_annealing_direct(
    m_candidates: int,
    m_facilities: int,
    max_iterations: int = 10000,
    initial_temp: float = 100.0,
    cooling_rate: float = 0.9
) -> Tuple[List[int], int]:
    """
    Direct simulated annealing without creating the full graph.
    Uses vote-to-vote distance computation on demand.

    Args:
        m_candidates: Number of candidates (votes are permutations of 0..m_candidates-1)
        m_facilities: Number of facilities to place
        max_iterations: Maximum number of iterations
        initial_temp: Initial temperature for annealing
        cooling_rate: Rate at which temperature decreases

    Returns:
        (facility_votes_as_integers, total_cost)
    """
    from itertools import permutations

    candidates = list(range(m_candidates))
    all_votes = list(permutations(candidates))   # ⚠ factorial growth!
    vote_to_int, int_to_vote = create_vote_integer_mapping(m_candidates)

    n = len(all_votes)
    all_vote_ints = list(range(n))

    if m_facilities > n:
        return all_vote_ints, 0

    # Initialize with random solution
    current_facilities = np.random.choice(n, size=m_facilities, replace=False).tolist()
    current_cost = compute_total_cost_direct(all_votes, current_facilities, vote_to_int, int_to_vote)

    best_facilities = current_facilities.copy()
    best_cost = current_cost

    temperature = initial_temp
    temp_update_freq = max(1, max_iterations // 1000)

    for iteration in range(max_iterations):
        # Neighbor: swap one facility
        new_facilities = current_facilities.copy()
        available = [i for i in all_vote_ints if i not in current_facilities]

        if available:
            facility_to_replace = random.randrange(m_facilities)
            new_node = random.choice(available)
            new_facilities[facility_to_replace] = new_node

            new_cost = compute_total_cost_direct(all_votes, new_facilities, vote_to_int, int_to_vote)
            cost_diff = new_cost - current_cost

            # Acceptance rule
            if cost_diff < 0:
                accept = True
            else:
                try:
                    accept = random.random() < math.exp(-cost_diff / max(temperature, 1e-9))
                except OverflowError:
                    accept = False

            if accept:
                current_facilities, current_cost = new_facilities, new_cost
                if current_cost < best_cost:
                    best_facilities, best_cost = current_facilities.copy(), current_cost

        # Cooling schedule
        if iteration % temp_update_freq == 0:
            temperature *= cooling_rate
            if temperature < 1e-2:
                break

    return best_facilities, int(best_cost)


def compute_total_cost_direct(all_votes: List[tuple], facilities: List[int],
                            vote_to_int: Dict[tuple, int], int_to_vote: Dict[int, tuple]) -> float:
    """
    Compute total cost directly from vote permutations without using graph.

    Args:
        all_votes: List of all vote permutations
        facilities: List of facility indices (integers representing votes)
        vote_to_int: Mapping from vote tuples to integers
        int_to_vote: Mapping from integers to vote tuples

    Returns:
        Total cost (sum of swap distances from each vote to closest facility)
    """
    if not facilities:
        return float('inf')

    facility_votes = [int_to_vote[f] for f in facilities]
    total_cost = 0

    for vote in all_votes:
        min_distance = min(swap_distance(vote, fac_vote) for fac_vote in facility_votes)
        total_cost += min_distance

    return total_cost

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

def compute_optimal_nodes(num_candidates, domain_sizes, method_name):
    results = []

    print("create a graph")
    vote_graph, vote_to_int, int_to_vote = create_vote_swap_graph(num_candidates)
    print("graph created")

    previous_nodes = []

    for domain_size in domain_sizes:
        print(f"Processing domain size: {domain_size}")

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
                vote_graph, domain_size, max_iterations=10000)
        else:
            raise ValueError(f"Unknown method: {method_name}")

        previous_nodes = optimal_nodes

        # Convert results for consistency
        if method_name == 'sa':
            vote_to_int, int_to_vote = create_vote_integer_mapping(num_candidates)

        # Store results
        result = {
            'domain_size': domain_size,
            'total_cost': total_cost,
            'optimal_nodes_int': str(optimal_nodes),
            'optimal_nodes_votes': str([int_to_vote[f] for f in optimal_nodes]),
        }
        results.append(result)

    # Save all results to file
    save_optimal_nodes_results(results, num_candidates, method_name)

def load_optimal_nodes_results(num_candidates: int, method_name: str) -> List[Dict]:
    """
    Load optimal nodes results from CSV file for a specific method.

    Args:
        num_candidates: Number of candidates used
        method_name: Method name ('ilp', 'lp', 'sa')

    Returns:
        List of dictionaries containing results for each domain size
    """
    csv_filename = f'data/optimal_nodes/{method_name}_m{num_candidates}.csv'

    results = []
    try:
        with open(csv_filename, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Convert string values back to appropriate types
                result = {
                    'domain_size': int(row['domain_size']),
                    'total_cost': int(row['total_cost']),
                    'optimal_nodes_int': eval(row['optimal_nodes_int']),
                    'optimal_nodes_votes': eval(row['optimal_nodes_votes'])
                }
                results.append(result)
        print(f"Loaded {len(results)} results for {method_name} from {csv_filename}")
    except FileNotFoundError:
        print(f"File {csv_filename} not found. Run computation for {method_name} first.")
    except Exception as e:
        print(f"Error loading file {csv_filename}: {e}")

    return results

def load_domain_size_data(num_candidates: int) -> Dict[str, List[float]]:
    """
    Load domain size data from CSV file.

    Args:
        num_candidates: Number of candidates used

    Returns:
        Dictionary mapping domain names to lists of diversity values
    """
    csv_filename = f'data/domain_size/domain_size_m{num_candidates}.csv'

    domain_data = {}
    try:
        with open(csv_filename, 'r') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)  # Skip header row
            domain_sizes = [int(x) for x in header[1:]]  # Extract domain sizes from header

            for row in reader:
                domain_name = row[0]
                values = [float(x) for x in row[1:]]
                domain_data[domain_name] = values

        print(f"Loaded domain size data for {len(domain_data)} domains from {csv_filename}")
    except FileNotFoundError:
        print(f"File {csv_filename} not found.")
    except Exception as e:
        print(f"Error loading domain size file: {e}")

    return domain_data



def plot_optimal_nodes_results(
        num_candidates: int,
        methods: List[str],
        with_structured_domains: bool = True):
    """
    Plot optimal nodes results using matplotlib.

    Args:
        num_candidates: Number of candidates used
        methods: List of method names to plot
        with_structured_domains: Whether to include structured domain lines
    """
    # Load results for each method
    all_results = {}
    for method in methods:
        results = load_optimal_nodes_results(num_candidates, method)
        if results:
            all_results[method] = results

    if not all_results:
        print("No results to display.")
        return

    # Calculate and print approximation ratios
    if 'ilp' in all_results:
        ilp_costs = {result['domain_size']: result['total_cost'] for result in all_results['ilp']}

        for method in methods:
            if method != 'ilp' and method in all_results:
                method_costs = {result['domain_size']: result['total_cost'] for result in all_results[method]}

                # Calculate approximation ratios for common domain sizes
                ratios = []
                for domain_size in ilp_costs:
                    if domain_size in method_costs and ilp_costs[domain_size] > 0:
                        ratio = method_costs[domain_size] / ilp_costs[domain_size]
                        ratios.append(ratio)

                if ratios:
                    avg_ratio = sum(ratios) / len(ratios)
                    print(f"Average approximation ratio for {method.upper()}: {avg_ratio:.4f}")
                else:
                    print(f"No valid approximation ratios found for {method.upper()}")

    # Load domain size data for horizontal lines
    if with_structured_domains:
        domain_data = load_domain_size_data(num_candidates)

    # Create the plot with larger font sizes
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 14})  # Set default font size

    # Define colors and styles for each method
    method_styles = {
        'ilp': {'marker': 'o', 'linestyle': '-', 'color': 'black', 'label': 'ILP (Optimal)', 'linewidth': 3, 'markersize': 5},
        'ilp_fast': {'marker': 'd', 'linestyle': '-', 'color': 'blue', 'label': 'LP', 'linewidth': 2, 'markersize': 4, 'alpha': 0.8},
        'sa': {'marker': 's', 'linestyle': '--', 'color': 'red', 'label': 'Simulated Annealing', 'linewidth': 2, 'markersize': 4, 'alpha': 0.8},
        'greedy_ilp': {'marker': '^', 'linestyle': '-', 'color': 'green', 'label': 'Greedy ILP', 'linewidth': 2, 'markersize': 4, 'alpha': 0.8},
        'greedy_ilp_fast': {'marker': 'v', 'linestyle': '-.', 'color': 'orange', 'label': 'Greedy LP', 'linewidth': 2, 'markersize': 4, 'alpha': 0.8}
    }

    # Plot results for each method
    for method, results in all_results.items():
        print(method)
        domain_sizes = [result['domain_size'] for result in results]
        total_costs = [result['total_cost'] for result in results]
        outer_diversity = [1 - tc / normalization(num_candidates) for tc in total_costs]

        # Add gray shading for constant increment regions (only for the first method)
        if method == list(all_results.keys())[0]:
            constant_regions = []
            for i in range(1, len(total_costs)):
                if total_costs[i-1] - total_costs[i] == 1:
                    constant_regions.append((domain_sizes[i-1], domain_sizes[i]))

            # Add gray shading for constant increment regions
            for start, end in constant_regions:
                plt.axvspan(start, end, alpha=0.3, color='gray', zorder=0)

        # Plot method results
        style = method_styles.get(method, {'marker': 'o', 'linestyle': '-', 'color': 'gray', 'label': method})
        plt.plot(domain_sizes, outer_diversity,
                marker=style['marker'],
                linestyle=style['linestyle'],
                color=style['color'],
                linewidth=style.get('linewidth', 2),
                markersize=style.get('markersize', 4),
                alpha=style.get('alpha', 1.0),
                label=style['label'])

    if with_structured_domains:
        # Add horizontal lines for domain size data
        if domain_data:
            for domain_name, values in domain_data.items():
                if len(values) > 0:
                    size = values
                    size_increase = [size[i] - size[i - 1] for i in range(1, len(size))]
                    total_diversity = 0
                    for i in range(len(values) - 1):
                        total_diversity += (i + 1) * size_increase[i]

                    total_diversity /= normalization(num_candidates)
                    total_diversity = 1 - total_diversity

                    domain_size = size[0]
                    horizontal_value = total_diversity

                    plt.axhline(y=horizontal_value, color=COLOR[domain_name],
                               linestyle='--', linewidth=2, label=LABEL[domain_name], alpha=0.8)

                    plt.scatter(domain_size, horizontal_value, color=COLOR[domain_name],
                                s=100, zorder=5)

    plt.xlabel('Domain Size', fontsize=22)
    plt.ylabel('Outer Diversity', fontsize=22)
    plt.title(f'Most Diverse Domain ({num_candidates} candidates)', fontsize=22)
    plt.grid(True, alpha=0.3)

    # Increase tick label sizes
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.ylim([-0.05, 1.05])

    # Move legend outside of the plot (left side) with larger font
    plt.legend(loc='lower right', fontsize=15)

    plt.tight_layout()

    plt.savefig(f'img/optimal_nodes/outer_diversity_{num_candidates}.png', dpi=200, bbox_inches='tight')
    plt.show()

