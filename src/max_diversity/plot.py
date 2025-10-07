import csv
from typing import Dict
from typing import List
from src.max_diversity.main import compute_optimal_nodes, compute_ic_threshold
from time import time
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import re

plt.rcParams['font.serif'] = ['Times New Roman']

from src.diversity.diversity_utils import normalization
from src.print_utils import COLOR, LABEL, SHORT_LABEL



from src.diversity.diversity_utils import normalization


def load_optimal_nodes_results(num_candidates: int, method_name: str) -> List[Dict]:
    """
    Load optimal nodes results from CSV file for a specific method.

    Args:
        num_candidates: Number of candidates used
        method_name: Method name ('ilp', 'lp', 'sa')

    Returns:
        List of dictionaries containing results for each domain size
    """
    csv_filename = f'data/optimal_nodes/{method_name}_m{num_candidates}.csv'

    results = []
    try:
        with open(csv_filename, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Convert string values back to appropriate types
                result = {
                    'domain_size': int(row['domain_size']),
                    'total_cost': float(row['total_cost']),
                    'optimal_nodes_int': eval(row['optimal_nodes_int']),
                    'optimal_nodes_votes': eval(row['optimal_nodes_votes'])
                }
                results.append(result)
        print(f"Loaded {len(results)} results for {method_name} from {csv_filename}")
    except FileNotFoundError:
        print(f"File {csv_filename} not found. Run computation for {method_name} first.")
    except Exception as e:
        print(f"Error loading file {csv_filename}: {e}")

    return results

def load_domain_size_data(num_candidates: int) -> Dict[str, List[float]]:
    """
    Load domain size data from CSV file.

    Args:
        num_candidates: Number of candidates used

    Returns:
        Dictionary mapping domain names to lists of diversity values
    """
    csv_filename = f'data/domain_size/domain_size_m{num_candidates}.csv'

    domain_data = {}
    try:
        with open(csv_filename, 'r') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)  # Skip header row
            domain_sizes = [int(x) for x in header[1:]]  # Extract domain sizes from header

            for row in reader:
                domain_name = row[0]
                values = [float(x) for x in row[1:]]
                domain_data[domain_name] = values

        print(f"Loaded domain size data for {len(domain_data)} domains from {csv_filename}")
    except FileNotFoundError:
        print(f"File {csv_filename} not found.")
    except Exception as e:
        print(f"Error loading domain size file: {e}")

    return domain_data



def plot_holy_ic(base, num_candidates, with_structured_domains=True, ax=None):

    if with_structured_domains:
        raw_domain_data = load_domain_size_data(num_candidates)

    domain_data = {}
    if with_structured_domains:
        for d in base:
            if d in raw_domain_data:
                domain_data[d] = raw_domain_data[d]


    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    print(domain_data)

    if with_structured_domains:
        # Add horizontal lines for domain size data
        if domain_data:
            for domain_name, values in domain_data.items():
                if len(values) > 0:
                    size = values
                    size_increase = [size[i] - size[i - 1] for i in range(1, len(size))]
                    total_diversity = 0
                    for i in range(len(values) - 1):
                        total_diversity += (i + 1) * size_increase[i]

                    total_diversity /= normalization(num_candidates)
                    total_diversity = 1 - total_diversity

                    domain_size = size[0]
                    horizontal_value = total_diversity

                    ax.axhline(y=horizontal_value, color=COLOR[domain_name],
                               linestyle='--', linewidth=2, label=LABEL[domain_name], alpha=0.8)

                    ax.scatter(domain_size, horizontal_value, color=COLOR[domain_name],
                                s=100, zorder=5)

    min_value = 5
    runs_range = range(10)
    threshold_range = range(min_value, 25 + 1)
    data_dir = os.path.join('data', 'threshold_ic')
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

    ax.scatter(domain_sizes, outer_diversities, alpha=0.8, marker='x', color='black')
    ax.set_xlabel('Domain Size', fontsize=36)
    ax.set_ylabel('Outer Diversity', fontsize=36)
    ax.tick_params(axis='both', which='major', labelsize=28)
    ax.tick_params(axis='both', which='minor', labelsize=28)

    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.savefig(f'images/threshold_ic_scatter_{min_value}.png', dpi=300)
    # plt.show()


