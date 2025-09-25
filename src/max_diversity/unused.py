# import networkx as nx
# from typing import Dict
#
# def vote_to_integer(vote: tuple, m: int) -> int:
#     """
#     Convert a vote (permutation) to a unique integer.
#     Uses factorial number system (Lehmer code).
#
#     Args:
#         vote: Tuple representing a permutation
#         m: Number of candidates
#
#     Returns:
#         Unique integer representing the vote
#     """
#     candidates = list(range(m))
#     result = 0
#     factorial = 1
#
#     for i in range(m-1, 0, -1):
#         pos = candidates.index(vote[i])
#         result += pos * factorial
#         candidates.pop(pos)
#         factorial *= (m - i + 1)
#
#     return result
#
#
# def integer_to_vote(code: int, m: int) -> tuple:
#     """
#     Convert an integer back to a vote (permutation).
#
#     Args:
#         code: Integer representing the vote
#         m: Number of candidates
#
#     Returns:
#         Tuple representing the permutation
#     """
#     candidates = list(range(m))
#     result = []
#
#     for i in range(m-1, 0, -1):
#         factorial = 1
#         for j in range(1, i+1):
#             factorial *= j
#
#         pos = code // factorial
#         result.append(candidates.pop(pos))
#         code %= factorial
#
#     result.append(candidates[0])
#     return tuple(reversed(result))
#
#
# def compute_single_scores(graph: nx.Graph, facilities):
#     """
#     Compute individual scores: distance from each node to closest facility.
#     """
#     nodes = list(graph.nodes())
#     scores = []
#
#     for node in nodes:
#         min_dist = float('inf')
#         for facility in facilities:
#             dist = nx.shortest_path_length(graph, node, facility)
#             min_dist = min(min_dist, dist)
#         scores.append(min_dist)
#
#     return scores
#
# def find_optimal_facilities_simulated_annealing_direct(
#     m_candidates: int,
#     m_facilities: int,
#     max_iterations: int = 10000,
#     initial_temp: float = 100.0,
#     cooling_rate: float = 0.9
# ) -> Tuple[List[int], int]:
#     """
#     Direct simulated annealing without creating the full graph.
#     Uses vote-to-vote distance computation on demand.
#
#     Args:
#         m_candidates: Number of candidates (votes are permutations of 0..m_candidates-1)
#         m_facilities: Number of facilities to place
#         max_iterations: Maximum number of iterations
#         initial_temp: Initial temperature for annealing
#         cooling_rate: Rate at which temperature decreases
#
#     Returns:
#         (facility_votes_as_integers, total_cost)
#     """
#     from itertools import permutations
#
#     candidates = list(range(m_candidates))
#     all_votes = list(permutations(candidates))   # ⚠ factorial growth!
#     vote_to_int, int_to_vote = create_vote_integer_mapping(m_candidates)
#
#     n = len(all_votes)
#     all_vote_ints = list(range(n))
#
#     if m_facilities > n:
#         return all_vote_ints, 0
#
#     # Initialize with random solution
#     current_facilities = np.random.choice(n, size=m_facilities, replace=False).tolist()
#     current_cost = compute_total_cost_direct(all_votes, current_facilities, vote_to_int, int_to_vote)
#
#     best_facilities = current_facilities.copy()
#     best_cost = current_cost
#
#     temperature = initial_temp
#     temp_update_freq = max(1, max_iterations // 1000)
#
#     for iteration in range(max_iterations):
#         # Neighbor: swap one facility
#         new_facilities = current_facilities.copy()
#         available = [i for i in all_vote_ints if i not in current_facilities]
#
#         if available:
#             facility_to_replace = random.randrange(m_facilities)
#             new_node = random.choice(available)
#             new_facilities[facility_to_replace] = new_node
#
#             new_cost = compute_total_cost_direct(all_votes, new_facilities, vote_to_int, int_to_vote)
#             cost_diff = new_cost - current_cost
#
#             # Acceptance rule
#             if cost_diff < 0:
#                 accept = True
#             else:
#                 try:
#                     accept = random.random() < math.exp(-cost_diff / max(temperature, 1e-9))
#                 except OverflowError:
#                     accept = False
#
#             if accept:
#                 current_facilities, current_cost = new_facilities, new_cost
#                 if current_cost < best_cost:
#                     best_facilities, best_cost = current_facilities.copy(), current_cost
#
#         # Cooling schedule
#         if iteration % temp_update_freq == 0:
#             temperature *= cooling_rate
#             if temperature < 1e-2:
#                 break
#
#     return best_facilities, int(best_cost)
#
#
# def compute_total_cost_direct(all_votes: List[tuple], facilities: List[int],
#                             vote_to_int: Dict[tuple, int], int_to_vote: Dict[int, tuple]) -> float:
#     """
#     Compute total cost directly from vote permutations without using graph.
#
#     Args:
#         all_votes: List of all vote permutations
#         facilities: List of facility indices (integers representing votes)
#         vote_to_int: Mapping from vote tuples to integers
#         int_to_vote: Mapping from integers to vote tuples
#
#     Returns:
#         Total cost (sum of swap distances from each vote to closest facility)
#     """
#     if not facilities:
#         return float('inf')
#
#     facility_votes = [int_to_vote[f] for f in facilities]
#     total_cost = 0
#
#     for vote in all_votes:
#         min_distance = min(swap_distance(vote, fac_vote) for fac_vote in facility_votes)
#         total_cost += min_distance
#
#     return total_cost