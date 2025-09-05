from src.optimal_votes import (
    compute_optimal_nodes,
    plot_optimal_nodes_results,
)

# Example usage
if __name__ == "__main__":

    methods = [
        # 'ilp',
        # 'ilp_fast',
        'sa',
        'greedy_ilp',
        'greedy_ilp_fast',
    ]

    # num_candidates = 3
    # domain_sizes = range(1,6+1)

    # num_candidates = 4
    # domain_sizes = range(1,24+1)

    num_candidates = 5
    domain_sizes = range(1,20+1)

    # num_candidates = 8
    # domain_sizes = range(1,9+1)


    for method_name in methods:
        print('Method:', method_name)
        compute_optimal_nodes(num_candidates, domain_sizes, method_name)

    plot_optimal_nodes_results(num_candidates, methods, with_structured_domains=False)
