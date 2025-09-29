from src.max_diversity.main import compute_optimal_nodes, compute_ic_threshold
from src.max_diversity.plot import plot_optimal_nodes_results
from time import time


# Example usage
if __name__ == "__main__":

    methods = [
        'smpl_holy_ic',
    ]

    num_candidates = 8

    max_iterations = None
    num_samples = 1000

    for run in range(3):
        print(f'Run {run}')
        for threshold in reversed(range(20,25+1)):
            print(f'Threshold: {threshold}')
            for method_name in methods:
                compute_ic_threshold(num_candidates, method_name, threshold=threshold,
                                      num_samples=num_samples, run=run)


