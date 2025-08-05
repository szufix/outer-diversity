from itertools import product

from src.domain.validator import validate_domain


@validate_domain
def group_separable_caterpillar_domain(num_candidates):

    def recursive(candidates):

        if len(candidates) == 1:
            return [[candidates[0]]]

        # mid = len(candidates) // 2
        left = candidates[:1]
        right = candidates[1:]

        # Recursively get all left and right subtree rankings
        left_rankings = recursive(left)
        right_rankings = recursive(right)

        all_rankings = []

        for l, r in product(left_rankings, right_rankings):
            all_rankings.append(l + r)  # left before right
            all_rankings.append(r + l)  # right before left (flip)

        return all_rankings

    candidates = list(range(num_candidates))
    return recursive(candidates)


@validate_domain
def group_separable_balanced_domain(num_candidates):

    def recursive(candidates):

        if len(candidates) == 1:
            return [[candidates[0]]]

        mid = len(candidates) // 2
        left = candidates[:mid]
        right = candidates[mid:]

        # Recursively get all left and right subtree rankings
        left_rankings = recursive(left)
        right_rankings = recursive(right)

        all_rankings = []

        for l, r in product(left_rankings, right_rankings):
            all_rankings.append(l + r)  # left before right
            all_rankings.append(r + l)  # right before left (flip)

        return all_rankings

    candidates = list(range(num_candidates))
    return recursive(candidates)
