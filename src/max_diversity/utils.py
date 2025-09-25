from collections import deque
from itertools import permutations

import networkx as nx


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



def create_vote_integer_mapping(m: int):
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

def compute_total_cost(graph: nx.Graph, facilities) -> int:
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