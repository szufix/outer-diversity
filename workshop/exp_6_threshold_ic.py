from src.max_diversity.main import compute_optimal_nodes, compute_ic_threshold
from src.max_diversity.plot import plot_optimal_nodes_results
from time import time
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import re


def compute():

    methods = [
        'smpl_holy_ic',
    ]

    num_candidates = 8

    max_iterations = None
    num_samples = 1000

    for run in range(10):
        print(f'Run {run}')
        for threshold in reversed(range(10,25+1)):
            print(f'Threshold: {threshold}')
            for method_name in methods:
                compute_ic_threshold(num_candidates, method_name, threshold=threshold,
                                      num_samples=num_samples, run=run)


def plot():
    min_value = 8
    runs_range = range(10)
    threshold_range = range(min_value, 25 + 1)
    data_dir = os.path.join(os.path.dirname(__file__), 'data', 'threshold_ic')
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    domain_sizes = []
    outer_diversities = []

    filtered_files = []
    for file in csv_files:
        # Expect filename pattern with run and threshold, e.g. ..._run{run}_threshold{threshold}.csv
        match = re.search(r'_t(\d+)_r(\d+)\.csv', os.path.basename(file))
        if match:
            threshold = int(match.group(1))
            run = int(match.group(2))
            if run in runs_range and threshold in threshold_range:
                filtered_files.append(file)

    for file in filtered_files:
        df = pd.read_csv(file)
        if 'domain_size' in df.columns and 'total_cost' in df.columns:
            domain_sizes.extend(df['domain_size'].values)
            outer_diversities.extend(df['total_cost'].values)
        else:
            for col in df.columns:
                if 'domain' in col:
                    domain_sizes.extend(df[col].values)
                if 'diversity' in col:
                    outer_diversities.extend(df[col].values)

    print(f"Loaded {len(domain_sizes)} data points from {len(filtered_files)} files.")

    plt.figure(figsize=(10, 6))
    plt.scatter(domain_sizes, outer_diversities, alpha=0.4)
    plt.xlabel('Domain Size', fontsize=32)
    plt.ylabel('Outer Diversity', fontsize=32)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.ylim([0, 1])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'images/threshold_ic_scatter_{min_value}.png', dpi=300)
    plt.show()



if __name__ == "__main__":
    
    # compute()
    plot()


