import networkx as nx

from max_diversity.utils import compute_total_cost


def find_optimal_facilities_bruteforce(graph: nx.Graph, m: int):
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
