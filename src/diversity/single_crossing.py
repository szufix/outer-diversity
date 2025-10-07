def distance_vote_single_crossing_domain(vote, domain, domain_obj):
    """
    Compute the (optionally normalized) distance from a vote to the single crossing domain.
    If normalize=True, returns a value in [0, 1].
    """
    num_candidates = len(vote)

    edge_swaps = domain_obj.edge_swaps

    # Convert input vote to position mapping for easier comparison
    vote_positions = {candidate: pos for pos, candidate in enumerate(vote)}

    min_distance = float('inf')

    # Check distance to each vote in the domain
    for domain_vote in domain:
        distance = compute_swap_distance(vote, domain_vote)
        min_distance = min(min_distance, distance)

    # Try traversing from each extreme vote
    for start_idx in [0, len(domain) - 1]:
        extreme_vote = domain[start_idx]
        current_distance = compute_swap_distance(vote, extreme_vote)
        min_distance = min(min_distance, current_distance)

        if start_idx == 0:
            # Forward traversal
            for i in range(len(edge_swaps)):
                swap_pair = edge_swaps[i]
                if swap_pair:
                    c1, c2 = swap_pair
                    # In forward direction: we're swapping c1 and c2 (c1 moves right, c2 moves left)
                    # Check if this swap moves us closer to the target vote
                    if vote_positions[c1] > vote_positions[c2]:
                        # Target has c2 before c1, so moving c1 right and c2 left helps
                        current_distance -= 1
                    else:
                        # Target has c1 before c2, so this swap hurts
                        current_distance += 1

                    min_distance = min(min_distance, current_distance)
        else:
            # Backward traversal - reverse the swaps
            for i in range(len(edge_swaps) - 1, -1, -1):
                swap_pair = edge_swaps[i]
                if swap_pair:
                    c1, c2 = swap_pair
                    # In backward direction: we're undoing the swap (c2 moves right, c1 moves left)
                    if vote_positions[c1] > vote_positions[c2]:
                        # Target has c2 before c1, undoing the swap hurts
                        current_distance += 1
                    else:
                        # Target has c1 before c2, undoing the swap helps
                        current_distance -= 1

                    min_distance = min(min_distance, current_distance)


    return min_distance


def compute_swap_distance(vote1, vote2):
    """
    Compute the swap distance (number of adjacent swaps) between two votes.
    """
    # Convert to permutation indices
    perm = [vote1.index(candidate) for candidate in vote2]

    # Count inversions (bubble sort distance)
    distance = 0
    n = len(perm)
    for i in range(n):
        for j in range(i + 1, n):
            if perm[i] > perm[j]:
                distance += 1

    return distance
