from src.optimal_votes import (
    compute_optimal_nodes,
    plot_optimal_nodes_results,
    merge_individual_results,
    check_individual_results_status,
)
from time import time


# Example usage
if __name__ == "__main__":

    methods = [
        'ilp',  # Use individual computation for ILP
        # 'ilp_fast',
        # 'greedy_ilp',
        # 'sa',
        # 'smpl_sa'
    ]

    # num_candidates = 4
    # domain_sizes = range(1,24+1)

    # num_candidates = 5
    # domain_sizes = range(1,120+1)

    num_candidates = 6
    domain_sizes = range(1,3+1)

    # # For ILP: check if individual results exist and merge them
    # ilp_available, ilp_missing = check_individual_results_status(num_candidates, 'ilp', list(domain_sizes))
    # if ilp_available:
    #     print(f"Found {len(ilp_available)} individual ILP results, merging...")
    #     merge_individual_results(num_candidates, 'ilp', ilp_available)
    #     methods.append('ilp')  # Add ILP to plotting methods
    # else:
    #     print("No individual ILP results found. Use exp_3_optimal_votes_individual.py to compute them.")

    x = []
    for method_name in methods:
        # if method_name == 'ilp':
        #     print(f'Method: {method_name} (using merged results)')
        #     continue  # Skip computation, already merged

        print('Method:', method_name)
        start = time()
        compute_optimal_nodes(num_candidates, domain_sizes, method_name)
        end = time()
        print(f'Time taken: {end - start} seconds')
        x.append(end - start)
    print(x)

    # Plot all methods including merged ILP results
    plot_methods = [m for m in methods if m in ['ilp', 'ilp_fast', 'greedy_ilp', 'sa']]
    plot_optimal_nodes_results(num_candidates, plot_methods, domain_sizes, with_structured_domains=False)
