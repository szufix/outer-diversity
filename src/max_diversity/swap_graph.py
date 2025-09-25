from typing import Dict
from typing import Tuple

import networkx as nx

from src.max_diversity.utils import create_vote_integer_mapping


def create_vote_swap_graph(m: int) -> Tuple[nx.Graph, Dict[tuple, int], Dict[int, tuple]]:
    """
    Create a graph representing all possible votes with edges between votes
    that are at distance of one swap (adjacent transposition).

    Args:
        m: Number of candidates

    Returns:
        Tuple of (graph, vote_to_int_mapping, int_to_vote_mapping)
    """
    from itertools import permutations

    candidates = list(range(m))
    all_votes = list(permutations(candidates))

    # Create bidirectional mappings
    vote_to_int, int_to_vote = create_vote_integer_mapping(m)

    # Create graph with integer node IDs
    graph = nx.Graph()
    graph.add_nodes_from(range(len(all_votes)))

    # Add edges between votes that differ by exactly one adjacent swap
    for i, vote1 in enumerate(all_votes):
        for j, vote2 in enumerate(all_votes):
            if i < j:  # Avoid duplicate edges since graph is undirected
                if is_one_swap_distance(vote1, vote2):
                    graph.add_edge(i, j)

    print(f"Created vote swap graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    return graph, vote_to_int, int_to_vote

def is_one_swap_distance(vote1: tuple, vote2: tuple) -> bool:
    """
    Check if two votes differ by exactly one adjacent swap.

    Args:
        vote1, vote2: Tuples representing permutations

    Returns:
        True if votes differ by exactly one adjacent transposition
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

    pos1, pos2 = differences[0], differences[1]

    # Check if positions are adjacent
    if abs(pos1 - pos2) != 1:
        return False

    # Check if elements are swapped
    return vote1[pos1] == vote2[pos2] and vote1[pos2] == vote2[pos1]
