from src.optimal_votes import (
    compute_optimal_nodes,
    plot_optimal_nodes_results,
)
from time import time


# Example usage
if __name__ == "__main__":

    methods = [
        # 'ilp',
        # 'ilp_fast',
        # 'greedy_ilp',
        # 'sa',
        'smpl_sa'
    ]


    # num_candidates = 4
    # domain_sizes = range(1,24+1)

    # num_candidates = 5
    # domain_sizes = range(1,120+1)

    num_candidates = 7
    domain_sizes = range(1,9+1)

    # num_candidates = 8
    # domain_sizes = range(1,9+1)


    x = []
    for method_name in methods:
        print('Method:', method_name)
        start = time()
        compute_optimal_nodes(num_candidates, domain_sizes, method_name)
        end = time()
        print(f'Time taken: {end - start} seconds')
        x.append(end - start)
    print(x)
    plot_optimal_nodes_results(num_candidates, methods, with_structured_domains=False)
