import numpy as np

from src.domain.validator import validate_domain
from src.domain.extenders import extend_to_reversed


@validate_domain
def single_crossing_domain(num_candidates):
    domain_size = int(num_candidates * (num_candidates - 1) / 2 + 1)
    domain = [list(range(num_candidates))]
    for line in range(1, domain_size):
        all_swap_indices = [
            (j, j + 1)
            for j in range(num_candidates - 1)
            if domain[line - 1][j] < domain[line - 1][j + 1]
        ]
        swap_indices = all_swap_indices[np.random.randint(0, len(all_swap_indices))]

        new_line = domain[line - 1].copy()
        new_line[swap_indices[0]] = domain[line - 1][swap_indices[1]]
        new_line[swap_indices[1]] = domain[line - 1][swap_indices[0]]
        domain.append(new_line)
    return domain


@validate_domain
def ext_single_crossing_domain(num_candidates):
    domain = single_crossing_domain(num_candidates)
    return extend_to_reversed(domain)