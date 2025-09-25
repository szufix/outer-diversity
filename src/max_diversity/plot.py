import csv
from typing import Dict
from typing import List

import matplotlib.pyplot as plt

from src.diversity.diversity_utils import normalization
from src.print_utils import COLOR, LABEL


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
                    'total_cost': int(row['total_cost']),
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

def plot_optimal_nodes_results(
        num_candidates: int,
        methods: List[str],
        domain_sizes,
        with_structured_domains: bool = True):
    """
    Plot optimal nodes results using matplotlib.

    Args:
        num_candidates: Number of candidates used
        methods: List of method names to plot
        with_structured_domains: Whether to include structured domain lines
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

    # Calculate and print approximation ratios
    if 'ilp' in all_results:
        ilp_costs = {result['domain_size']: result['total_cost'] for result in all_results['ilp']}
        optimal_diversity = {key: 1 - tc / normalization(num_candidates) for key, tc in ilp_costs.items()}

        for method in methods:
            if method != 'ilp' and method in all_results:
                method_costs = {result['domain_size']: result['total_cost'] for result in all_results[method]}
                outer_diversity = {key: 1 - tc / normalization(num_candidates) for key, tc in method_costs.items()}

                # Calculate approximation ratios for common domain sizes
                # ratios = []
                # for domain_size in domain_sizes:
                #     if domain_size in method_costs and ilp_costs[domain_size] > 0:
                #         # print(method)
                #         # print("->", optimal_diversity[5])
                #         print(domain_size)
                #         print(outer_diversity[domain_size])
                #         print(optimal_diversity[domain_size])
                #         ratio = outer_diversity[domain_size] / optimal_diversity[domain_size]
                #         ratios.append(ratio)
                #
                # if ratios:
                #     avg_ratio = sum(ratios) / len(ratios)
                #     print(f"Average approximation ratio for {method.upper()}: {avg_ratio:.4f}")
                # else:
                #     print(f"No valid approximation ratios found for {method.upper()}")

    # Load domain size data for horizontal lines
    if with_structured_domains:
        domain_data = load_domain_size_data(num_candidates)

    # Create the plot with larger font sizes
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 14})  # Set default font size

    # Define colors and styles for each method
    method_styles = {
        'ilp': {'marker': 'o', 'linestyle': '-', 'color': 'black', 'label': 'ILP (Optimal)', 'linewidth': 3, 'markersize': 5},
        'greedy_ilp': {'marker': 'd', 'linestyle': '-', 'color': 'blue', 'label': 'Greedy ILP', 'linewidth': 2, 'markersize': 4, 'alpha': 0.8},
        'sa': {'marker': 's', 'linestyle': '--', 'color': 'red', 'label': 'SA', 'linewidth': 2, 'markersize': 4, 'alpha': 0.8},
        'smpl_sa': {'marker': '^', 'linestyle': '-', 'color': 'green', 'label': 'Sampling SA', 'linewidth': 2, 'markersize': 4, 'alpha': 0.8},
    }

    # Plot results for each method
    for method, results in all_results.items():
        print(method)
        domain_sizes = [result['domain_size'] for result in results]
        total_costs = [result['total_cost'] for result in results]
        outer_diversity = [1 - tc / normalization(num_candidates) for tc in total_costs]

        # Add gray shading for constant increment regions (only for the first method)
        if method == list(all_results.keys())[0]:
            constant_regions = []
            for i in range(1, len(total_costs)):
                if total_costs[i-1] - total_costs[i] == 1:
                    constant_regions.append((domain_sizes[i-1], domain_sizes[i]))

            # Add gray shading for constant increment regions
            for start, end in constant_regions:
                plt.axvspan(start, end, alpha=0.3, color='gray', zorder=0)

        # Plot method results
        style = method_styles.get(method, {'marker': 'o', 'linestyle': '-', 'color': 'gray', 'label': method})
        plt.plot(domain_sizes, outer_diversity,
                marker=style['marker'],
                linestyle=style['linestyle'],
                color=style['color'],
                linewidth=style.get('linewidth', 2),
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

                    plt.axhline(y=horizontal_value, color=COLOR[domain_name],
                               linestyle='--', linewidth=2, label=LABEL[domain_name], alpha=0.8)

                    plt.scatter(domain_size, horizontal_value, color=COLOR[domain_name],
                                s=100, zorder=5)

    plt.xlabel('Domain Size', fontsize=22)
    plt.ylabel('Outer Diversity', fontsize=22)
    plt.title(f'Most Diverse Domain ({num_candidates} candidates)', fontsize=22)
    plt.grid(True, alpha=0.3)

    # Increase tick label sizes
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.ylim([-0.05, 1.05])

    # Move legend outside of the plot (left side) with larger font
    plt.legend(loc='lower right', fontsize=15)

    plt.tight_layout()

    plt.savefig(f'images/optimal_nodes/outer_diversity_{num_candidates}.png', dpi=200, bbox_inches='tight')
    plt.show()
