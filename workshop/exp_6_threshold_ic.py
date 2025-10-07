import glob
import os
import re

import matplotlib.pyplot as plt
import pandas as pd

from src.max_diversity.main import compute_ic_threshold

plt.rcParams['font.serif'] = ['Times New Roman']

from src.print_utils import SHORT_LABEL

from src.diversity.diversity_utils import normalization
from src.print_utils import COLOR

from src.max_diversity.plot import load_domain_size_data, load_optimal_nodes_results


def plot_holy_ic_and_optimal_nodes_results(
            base,
            num_candidates: int,
            methods: list,
            domain_sizes,
            with_structured_domains: bool = True,
            ax=None):
        """
        Plot optimal nodes results using matplotlib.

        Args:
            num_candidates: Number of candidates used
            methods: List of method names to plot
            with_structured_domains: Whether to include structured domain lines
            ax: Optional matplotlib axes to plot on
        """
        # Load results for each method
        all_results = {}
        for method in methods:
            print(method)
            results = load_optimal_nodes_results(num_candidates, method)
            if results:
                all_results[method] = results[domain_sizes[0] - 1:domain_sizes[-1]]

        if not all_results:
            print("No results to display.")
            return


        if with_structured_domains:
            raw_domain_data = load_domain_size_data(num_candidates)

        domain_data = {}
        if with_structured_domains:
            for d in base:
                if d in raw_domain_data:
                    domain_data[d] = raw_domain_data[d]

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 12))

        # Define colors and styles for each method
        method_styles = {
            'ilp': {'marker': 'o', 'linestyle': '-', 'color': 'black', 'label': 'ILP (Optimal)',
                    'linewidth': 3, 'markersize': 5},
            'greedy_ilp': {'marker': 'd', 'linestyle': '-', 'color': 'blue', 'label': 'Greedy ILP',
                           'linewidth': 2, 'markersize': 4, 'alpha': 0.8},
            'sa': {'marker': 's', 'linestyle': '--', 'color': 'red', 'label': '~Max', 'linewidth': 2,
                   'markersize': 4, 'alpha': 0.8},
            'ic': {'marker': '^', 'linestyle': '-', 'color': 'blue', 'label': 'IC',
                   'linewidth': 2, 'markersize': 4, 'alpha': 0.8},
            'smpl_sa': {'marker': '^', 'linestyle': '-', 'color': 'black', 'label': '~Max',
                        'linewidth': 2, 'markersize': 4, 'alpha': 0.8},
            'smpl_ic': {'marker': '^', 'linestyle': '-', 'color': 'green', 'label': 'IC',
                        'linewidth': 2, 'markersize': 4, 'alpha': 0.8},
            'smpl_holy_ic': {'marker': '^', 'linestyle': '-', 'color': 'blue',
                             'label': 'Smpl Holy IC', 'linewidth': 2, 'markersize': 4,
                             'alpha': 0.8},
        }


        # Plot results for each method
        for method, results in all_results.items():
            print(method)
            domain_sizes = [result['domain_size'] for result in results]
            total_costs = [result['total_cost'] for result in results]
            # outer_diversity = [1 - tc / normalization(num_candidates) for tc in total_costs]
            outer_diversity = total_costs
            # Add gray shading for constant increment regions (only for the first method)
            if method == list(all_results.keys())[0]:
                constant_regions = []
                for i in range(1, len(total_costs)):
                    if total_costs[i - 1] - total_costs[i] == 1:
                        constant_regions.append((domain_sizes[i - 1], domain_sizes[i]))

                # Add gray shading for constant increment regions
                for start, end in constant_regions:
                    ax.axvspan(start, end, alpha=0.3, color='gray', zorder=0)

            # Plot method results
            style = method_styles.get(method, {'marker': 'o', 'linestyle': '-', 'color': 'gray',
                                               'label': method})
            ax.plot(domain_sizes, outer_diversity,
                    # marker=style['marker'],
                    linestyle=style['linestyle'],
                    color=style['color'],
                    linewidth=2,  # style.get('linewidth', 2),
                    # markersize=style.get('markersize', 4),
                    alpha=style.get('alpha', 1.0),
                    label=style['label'])

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

        ax.scatter(domain_sizes, outer_diversities,
                   alpha=0.8, marker='x', color='red', label='Thres.-IC', zorder=10)

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
                                   linestyle='--', linewidth=2, alpha=0.8)

                        ax.scatter(domain_size, horizontal_value, color=COLOR[domain_name],
                                   s=200, zorder=5, label=SHORT_LABEL[domain_name])

        ax.set_xlabel('Domain Size', fontsize=36)
        ax.set_ylabel('Outer Diversity', fontsize=36)
        ax.tick_params(axis='both', which='major', labelsize=28)
        ax.tick_params(axis='both', which='minor', labelsize=28)

        ax.grid(True, alpha=0.3)

        ax.set_ylim([-0.01, 1.01])
        ax.set_xlim([-5, 525])

        from matplotlib.lines import Line2D

        # After plotting everything
        handles, labels = ax.get_legend_handles_labels()

        # Replace the Thres.-IC handle with a custom one
        custom_handles = []
        for h, l in zip(handles, labels):
            if l == 'Thres.-IC':
                custom_handles.append(
                    Line2D([0], [0], marker='x', color='red', linestyle='None', markersize=16,
                           label='Thres.-IC'))
            else:
                custom_handles.append(h)

        ax.legend(custom_handles, labels, loc='lower right', fontsize=27, ncol=2)

        plt.tight_layout()

        plt.savefig(f'images/optimal_nodes/outer_diversity_{num_candidates}_with_hic.png', dpi=200,
                    bbox_inches='tight')
        plt.show()


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





if __name__ == "__main__":

    num_candidates = 8

    base = [
        # 'euclidean_3d',
        'euclidean_2d',
        'spoc',
        'sp_double_forked',
        'caterpillar',
        'balanced',
        'largest_condorcet',
        'single_peaked',
        'euclidean_1d',
        'single_crossing',
        # 'ext_single_vote',
        # 'single_vote',
    ]


    methods = [
        'smpl_sa',
        'smpl_ic',
    ]

    domain_sizes = range(1,520+1)

    # max_iterations = None
    # num_samples = 1000

    plot_holy_ic_and_optimal_nodes_results(base,
        num_candidates,
        methods,
        domain_sizes,
        with_structured_domains = True)

