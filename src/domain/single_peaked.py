from src.domain.validator import validate_domain
from src.domain.extenders import extend_to_reversed

@validate_domain
def single_peaked_domain(num_candidates):
    def recursive(a, b, rank, position):
        if a == b:
            rank[position] = a
            all_sp_ranks.append(rank[:])  # Make a shallow copy only when appending
            return
        rank[position] = a
        recursive(a + 1, b, rank, position - 1)

        rank[position] = b
        recursive(a, b - 1, rank, position - 1)

    all_sp_ranks = []
    recursive(0, num_candidates - 1, [0] * num_candidates, num_candidates - 1)

    return all_sp_ranks

@validate_domain
def spoc_domain(num_candidates):
    base_domain = single_peaked_domain(num_candidates)
    domain = []

    for i in range(num_candidates):
        for vote in base_domain:
            rotated = [(c + i) % num_candidates for c in vote]
            if rotated not in domain:
                domain.append(rotated)

    return domain

@validate_domain
def ext_single_peaked_domain(num_candidates):
    domain = single_peaked_domain(num_candidates)
    return extend_to_reversed(domain)
