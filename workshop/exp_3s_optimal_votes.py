
import sys
import argparse

from src.optimal_votes import (
    compute_optimal_nodes,
    plot_optimal_nodes_results,
)

def parse_args():
    parser = argparse.ArgumentParser(description='Bayesian model for the resampling distribution.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--method', type=str)

    parser.add_argument('--range', type=int)

    parser.add_argument('--m', type=int)

    args = parser.parse_args()
    return args


# Example usage
if __name__ == "__main__":

    args = parse_args()

    num_candidates = args.m
    method_name = args.method
    domain_sizes = range(1,args.range+1)


    compute_optimal_nodes(num_candidates, domain_sizes, method_name)
