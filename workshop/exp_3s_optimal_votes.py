
import sys
import argparse
from src.max_diversity.main import compute_optimal_nodes
from src.max_diversity.plot import plot_optimal_nodes_results

def parse_args():
    parser = argparse.ArgumentParser(description='...',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--method', type=str)

    parser.add_argument('--range', type=int)

    parser.add_argument('--m', type=int)

    parser.add_argument('--iter', type=int)

    parser.add_argument('--samples', type=int)

    args = parser.parse_args()
    return args

# nohup python -u exp_3s_optimal_votes.py --method sa --range 20 --m 5 --iter 10 --samples 100 > logs/exp_3s_optimal_votes_m5_sa.log &

# Example usage
if __name__ == "__main__":

    args = parse_args()

    num_candidates = args.m
    method_name = args.method
    domain_sizes = range(1,args.range+1)

    max_iterations = args.iter
    num_samples = args.samples

    compute_optimal_nodes(num_candidates, domain_sizes, method_name, max_iterations, num_samples)
