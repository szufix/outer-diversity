from src.max_diversity.main import compute_optimal_nodes
from src.max_diversity.plot import plot_optimal_nodes_results
from time import time


# Example usage
if __name__ == "__main__":

    methods = [
        # 'ilp',  # Use individual computation for ILP
        # 'greedy_ilp',
        'sa',
        'smpl_sa'
        # 'ic',
        # 'smpl_ic'
    ]

    # num_candidates = 4
    # domain_sizes = range(1,24+1)

    # num_candidates = 5
    # domain_sizes = range(1,120+1)

    # num_candidates = 6
    # domain_sizes = range(1,120+1)

    num_candidates = 5
    domain_sizes = range(1,10+1)

    max_iterations = 100
    num_samples = 100


    x = []
    for method_name in methods:
        # if method_name == 'ilp':
        #     print(f'Method: {method_name} (using merged results)')
        #     continue  # Skip computation, already merged

        print('Method:', method_name)
        start = time()
        compute_optimal_nodes(num_candidates, domain_sizes, method_name,
                              num_samples=num_samples, max_iterations=max_iterations)
        end = time()
        print(f'Time taken: {end - start} seconds')
        x.append(end - start)
    print(x)

    # plot_optimal_nodes_results(num_candidates, methods, domain_sizes, with_structured_domains=True)
