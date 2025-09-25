import networkx as nx


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


def compute_single_scores(graph: nx.Graph, facilities):
    """
    Compute individual scores: distance from each node to closest facility.
    """
    nodes = list(graph.nodes())
    scores = []

    for node in nodes:
        min_dist = float('inf')
        for facility in facilities:
            dist = nx.shortest_path_length(graph, node, facility)
            min_dist = min(min_dist, dist)
        scores.append(min_dist)

    return scores