from typing import List, Tuple

import gurobipy as gp
import networkx as nx
from gurobipy import GRB

from src.max_diversity.utils import compute_total_cost


def find_optimal_facilities_greedy_ilp(
        graph: nx.Graph, m: int, previous_nodes):
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
        graph: nx.Graph, m: int, previos_nodes):
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




def find_optimal_ilp(graph: nx.Graph, m: int):
    """
    Solve the facility location problem via ILP with enhanced debugging.
    """
    nodes = list(graph.nodes())
    n = len(nodes)

    # Handle edge cases
    if m >= n:
        return nodes, 0
    if m <= 0:
        return [], compute_total_cost(graph, set())

    # Precompute all-pairs shortest path distances
    distances = {i: nx.single_source_shortest_path_length(graph, i) for i in nodes}

    # Create model
    model = gp.Model("compact_facility_location")
    model.setParam("OutputFlag", 0)
    model.setParam("Threads", 0)       # Use all threads
    model.setParam("MIPGap", 1e-3)
    model.setParam('Cuts', 2)          # Aggressive cuts
    model.Params.TimeLimit = 48 * 60 * 60  # 172800 seconds

    # Decision variables
    x = model.addVars(nodes, vtype=GRB.BINARY, name="facility")
    y = model.addVars(nodes, nodes, vtype=GRB.BINARY, name="assign")

    # Objective: minimize total distance
    obj = gp.quicksum(distances[j][i] * y[i, j] for i in nodes for j in nodes)
    model.setObjective(obj, GRB.MINIMIZE)

    # Constraints
    model.addConstrs((gp.quicksum(y[i, j] for j in nodes) == 1 for i in nodes), name="serve")
    model.addConstrs((y[i, j] <= x[j] for i in nodes for j in nodes), name="open")
    model.addConstr(gp.quicksum(x[j] for j in nodes) == m, name="facility_count")

    # Warm start with greedy heuristic
    try:
        greedy_facilities, greedy_cost = find_optimal_facilities_greedy(graph, m)
        print(f"ILP warm start with greedy solution: cost={greedy_cost}")

        for j in nodes:
            x[j].start = int(j in greedy_facilities)
        for i in nodes:
            closest = min(greedy_facilities, key=lambda f: distances[f][i])
            for j in nodes:
                y[i, j].start = int(j == closest)
    except Exception as e:
        print(f"Warm start failed: {e}")

    # Solve
    model.optimize()

    def extract_solution():
        facilities = [j for j in nodes if x[j].X > 0.5]

        # Verify the cost calculation
        model_cost = int(model.objVal)
        return facilities, model_cost

    if model.status == GRB.OPTIMAL:
        facilities, actual_cost = extract_solution()
        print(f"ILP optimal solution for m={m}: cost={actual_cost}, gap={model.MIPGap:.6f}, status=OPTIMAL")
        return facilities, actual_cost
    elif model.status == GRB.TIME_LIMIT and model.SolCount > 0:
        facilities, actual_cost = extract_solution()
        print(f"ILP time limit reached for m={m}. Best: {actual_cost}, gap={model.MIPGap:.6f}")
        return facilities, actual_cost

    print(f"ILP failed with status {model.status} for m={m}")


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
    model.setParam('Heuristics', 0.2)  # Spend 20% time on heuristics
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


