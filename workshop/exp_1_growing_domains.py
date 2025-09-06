
import matplotlib.pyplot as plt
from src.print_utils import *
from src.diversity.diversity_utils import *
from src.diversity.growing_domains import *


def plot_domain_size_total(base, num_candidates) -> None:
    """Plots domain sizes from CSV data."""

    size = load_domain_size_csv(num_candidates)

    plt.figure(figsize=(4.8, 6.4))

    max_num_swaps = get_max_num_swaps(num_candidates)
    x_range = list(range(0, max_num_swaps + 1))

    for name in base:
        if name in size:
            plt.plot(x_range,
                     size[name],
                     label=LABEL[name],
                     marker=MARKER[name],
                     color=COLOR[name])
        else:
            print(f"Warning: {name} not found in the CSV data.")

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Number of swaps', fontsize=22)
    plt.ylabel('Domain size', fontsize=22)
    plt.title(f'Total size for {num_candidates} candidates', fontsize=22)
    plt.legend(fontsize=18, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig(f'img/domain_size_m{num_candidates}_total.png', dpi=200, bbox_inches='tight')
    plt.close()


def plot_domain_size_increase(base, num_candidates) -> None:
    """Plots domain sizes from CSV data."""

    size = load_domain_size_csv(num_candidates)

    max_num_swaps = get_max_num_swaps(num_candidates)
    x_range = list(range(0, max_num_swaps + 1))

    # Calculate the increase in domain size
    size_increase = compute_balls_increase(size)

    plt.figure(figsize=(6.4, 4.8))

    for name in base:
        if name in size:
            plt.plot(x_range[1:],  # Skip the first x_range value for increase
                     size_increase[name],
                     label=LABEL[name],
                     marker=MARKER[name],
                     color=COLOR[name])
        else:
            print(f"Warning: {name} not found in the CSV data.")
    # add text labels on x axis only for 1,2,3,4,5
    plt.xticks(x_range[1:], fontsize=16)  # Skip the first value for increase

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Number of swaps', fontsize=22)
    plt.ylabel('Size increase', fontsize=22)
    plt.title(f'DS Increase for {num_candidates} candidates', fontsize=22)
    plt.legend(fontsize=18, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig(f'img/domain_size_m{num_candidates}_increase.png', dpi=200, bbox_inches='tight')
    plt.close()


def print_domain_diversity(base, candidate_range):

    # Store all diversity values
    diversity_data = {}
    for num_candidates in candidate_range:
        size = load_domain_size_csv(num_candidates)

        max_num_swaps = get_max_num_swaps(num_candidates)
        x_range = list(range(0, max_num_swaps + 1))

        # Calculate the increase in domain size
        size_increase = compute_balls_increase(size)

        diversity_data[num_candidates] = {}

        for name in base:
            if name not in size_increase:
                print(f"Warning: {name} not found in the CSV data.")
                return

            total_diversity = 0
            # Calculate total diversity across all domains
            # which is equal to 1*[0] + 2*[1] + 3*[2] + 4*[3] + 5*[4]
            for i in range(len(x_range)-1):
                total_diversity += (i+1) * size_increase[name][i]
                print(i, total_diversity)

            # print(total_diversity, normalization(num_candidates))
            total_diversity /= normalization(num_candidates)
            total_diversity = 1 - total_diversity

            diversity_data[num_candidates][name] = total_diversity

    # Sort base by diversity values in the first column
    first_candidate = candidate_range[0]
    sorted_base = sorted(base, key=lambda name: diversity_data[first_candidate][name])

    # Print LaTeX table
    print("\\begin{tabular}{c" + "c" * len(candidate_range) + "}")
    print("\\hline")

    # Header row
    header = "Domain"
    for num_candidates in candidate_range:
        header += f" & {num_candidates}"
    header += " \\\\"
    print(header)
    print("\\hline")

    # Data rows
    for name in sorted_base:
        row = LABEL[name]
        for num_candidates in candidate_range:
            row += f" & {round(diversity_data[num_candidates][name], 3)}"
        row += " \\\\"
        print(row)

    print("\\hline")
    print("\\end{tabular}")




if __name__ == "__main__":

    base = [
        # 'euclidean_3d',
        # 'euclidean_2d',
        # 'spoc',
        # 'sp_double_forked',
        # 'caterpillar',
        # 'balanced',
        # 'single_peaked',
        # 'single_crossing',
        # 'euclidean_1d',
        # 'ext_single_vote',
        'single_vote',
    ]

    candidate_range = [3,4,5,6]

    for num_candidates in candidate_range:
        compute_domain_balls(base, num_candidates)

        # plot_domain_size_total(base, num_candidates)
        # plot_domain_size_increase(base, num_candidates)

    print_domain_diversity(base, candidate_range)


