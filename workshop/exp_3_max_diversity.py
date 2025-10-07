import matplotlib.pyplot as plt

from diversity.diversity_utils import normalization
from print_utils import *
from src.max_diversity.plot import load_optimal_nodes_results, load_domain_size_data


def plot_optimal_nodes_results(
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
            all_results[method] = results[domain_sizes[0]-1:domain_sizes[-1]]

    if not all_results:
        print("No results to display.")
        return

    # Load domain size data for horizontal lines
    if with_structured_domains:
        domain_data = load_domain_size_data(num_candidates)

    # remove 'euclidean_3d' if it exists
    if with_structured_domains and 'euclidean_3d' in domain_data:
        del domain_data['euclidean_3d']

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 10))

        # Define colors and styles for each method
        method_styles = {
            'ilp': {'marker': 'o', 'linestyle': '-', 'color': 'red', 'label': 'ILP (Optimal)',
                    'linewidth': 3, 'markersize': 5},
            'greedy_ilp': {'marker': 'd', 'linestyle': '-', 'color': 'blue', 'label': 'Greedy ILP',
                           'linewidth': 2, 'markersize': 4, 'alpha': 0.8},
            'sa': {'marker': 's', 'linestyle': '-', 'color': 'black', 'label': '~Max',
                   'linewidth': 2,
                   'markersize': 4, 'alpha': 0.8},
            'ic': {'marker': '^', 'linestyle': '-', 'color': 'green', 'label': 'IC',
                   'linewidth': 2, 'markersize': 4, 'alpha': 0.8},
            'smpl_sa': {'marker': '^', 'linestyle': '-', 'color': 'black', 'label': '~Max',
                        'linewidth': 2, 'markersize': 4, 'alpha': 0.8},
            'smpl_ic': {'marker': '^', 'linestyle': '-', 'color': 'green', 'label': 'IC',
                        'linewidth': 2, 'markersize': 4, 'alpha': 0.8},
            'smpl_holy_ic': {'marker': '^', 'linestyle': '-', 'color': 'blue',
                             'label': 'Smpl Holy IC', 'linewidth': 2, 'markersize': 4,
                             'alpha': 0.8},
        }

    # normalize the values for 'sa'
    import math
    n = math.factorial(num_candidates)
    max_cost = normalization(num_candidates)
    if 'sa' in all_results:
        for result in all_results['sa']:
            result['total_cost'] = 1 - result['total_cost'] / n * 2 / math.comb(num_candidates, 2)

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
                if total_costs[i-1] - total_costs[i] == 1:
                    constant_regions.append((domain_sizes[i-1], domain_sizes[i]))

            # Add gray shading for constant increment regions
            for start, end in constant_regions:
                ax.axvspan(start, end, alpha=0.3, color='gray', zorder=0)

        # Plot method results
        style = method_styles.get(method, {'marker': 'o', 'linestyle': '-', 'color': 'gray', 'label': method})
        ax.plot(domain_sizes, outer_diversity,
                # marker=style['marker'],
                linestyle=style['linestyle'],
                color=style['color'],
                linewidth=2, #style.get('linewidth', 2),
                markersize=style.get('markersize', 4),
                alpha=style.get('alpha', 1.0),
                label=style['label'])

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

    ax.set_xlabel('Domain Size', fontsize=28)
    ax.set_ylabel('Outer Diversity', fontsize=28)
    ax.grid(True, alpha=0.3)

    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=22)
    ax.tick_params(axis='both', which='minor', labelsize=22)

    ax.set_ylim([-0.01, 1.01])

    # Move legend outside of the plot (left side) with larger font
    ax.legend(loc='lower right', fontsize=27)

    plt.tight_layout()

    plt.savefig(f'images/optimal_nodes/outer_diversity_{num_candidates}.png', dpi=200, bbox_inches='tight')
    plt.show()


# Example usage
if __name__ == "__main__":

    methods = [
        'ilp',  # Use individual computation for ILP
        'sa',
        'ic',
    ]

    num_candidates = 6
    domain_sizes = range(1,30+1)

    # num_candidates = 8
    # domain_sizes = range(1,800+1)

    max_iterations = None
    num_samples = 1000

    # x = []
    # for method_name in methods:
    #     # if method_name == 'ilp':
    #     #     print(f'Method: {method_name} (using merged results)')
    #     #     continue  # Skip computation, already merged
    #
    #     print('Method:', method_name)
    #     start = time()
    #     compute_optimal_nodes(num_candidates, domain_sizes, method_name,
    #                           num_samples=num_samples, max_iterations=max_iterations)
    #     end = time()
    #     print(f'Time taken: {end - start} seconds')
    #     x.append(end - start)
    # print(x)

    plot_optimal_nodes_results(num_candidates, methods, domain_sizes,
                               with_structured_domains=False)
