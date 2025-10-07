from src.domain import *
from src.domain.extenders import *

domains = {
    'single_vote': single_vote_domain,
    'ext_single_vote': ext_single_vote_domain,
}

def compute_size_increase(size_data):
    """Helper function to compute size increase from size data."""
    size_increase = {}
    for name in size_data:
        size_increase[name] = [size_data[name][i] - size_data[name][i-1] for i in range(1, len(size_data[name]))]
    return size_increase

def compute_domain_size(base, num_candidates, x_range) -> None:
    """Computes domain sizes for given parameters and saves to CSV."""
    size = {name: [] for name in base}

    for name in base:

        domain = domains[name](num_candidates=num_candidates)

        for _ in x_range:
            domain = extend_by_swaps(domain, 1)
            size[name].append(len(domain))

    size_increase = compute_size_increase(size)

    diversity_data = {}

    for name in base:
        if name not in size_increase:
            print(f"Warning: {name} not found in the CSV data.")
            return

        total_diversity = 0
        # Calculate total diversity across all domains
        # which is equal to 1*[0] + 2*[1] + 3*[2] + 4*[3] + 5*[4]
        for i in range(len(x_range) - 1):
            total_diversity += (i + 1) * size_increase[name][i]

        diversity_data[name] = total_diversity

    # print(size)
    print(size_increase)
    print(diversity_data)


if __name__ == "__main__":

    base = [
        'single_vote',
        # 'ext_single_vote',
    ]

    candidate_range = [5,6,7]
    x_range = range(0, 9 + 1)

    for num_candidates in candidate_range:
        m = num_candidates
        max_value = m*(m-1) / 2
        print(f"Computing domain sizes for {num_candidates} candidates with max swaps {max_value}...")
        compute_domain_size(base, num_candidates, x_range)