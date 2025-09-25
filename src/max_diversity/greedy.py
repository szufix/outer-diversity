import networkx as nx
from src.max_diversity.utils import compute_total_cost, bfs_distances
import networkx as nx



def find_optimal_facilities_greedy(graph: nx.Graph, m: int):
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
