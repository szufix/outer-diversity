import matplotlib.pyplot as plt
import csv
import os

from src.domain.extenders import *
from src.domain import *
from src.print_utils import *

from src.outer_diversity import normalization

domains = {
    'euclidean_3d': euclidean_3d_domain,
    'euclidean_2d': euclidean_2d_domain,
    'euclidean_1d': euclidean_1d_domain,
    'caterpillar': group_separable_caterpillar_domain,
    'balanced': group_separable_balanced_domain,
    'single_peaked': single_peaked_domain,
    'single_crossing': single_crossing_domain,
    'sp_double_forked': sp_double_forked_domain,
    'spoc': spoc_domain,
    'ext_single_vote': ext_single_vote_domain,
    'single_vote': single_vote_domain,
}




def load_domain_size_csv(num_candidates):
    csv_filename = f'data/domain_size_m{num_candidates}.csv'

    size_data = {}
    with open(csv_filename, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header

        for row in reader:
            domain_name = row[0]
            sizes = [int(size) for size in row[1:]]
            size_data[domain_name] = sizes

    return size_data


def _get_max_num_swaps(m):
    return int(m*(m-1)/2 + 1)



def save_domain_size_csv(size, num_candidates):

    max_num_swaps = _get_max_num_swaps(num_candidates)
    x_range = list(range(0, max_num_swaps + 1))

    os.makedirs('data', exist_ok=True)
    csv_filename = f'data/domain_size_m{num_candidates}.csv'

    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['domain'] + list(x_range))
        for name in size:
            writer.writerow([name] + size[name])



def compute_domain_size(base, num_candidates) -> None:
    """Computes domain sizes for given parameters and saves to CSV."""
    size = {name: [] for name in base}

    for name in base:

        domain = domains[name](num_candidates=num_candidates)

        size[name].append(len(domain))

        max_num_swaps = _get_max_num_swaps(num_candidates)

        for _ in range(1, max_num_swaps+1):
            if len(size[name]) >= 2  and size[name][-1] == size[name][-2]: # no increase in size
                size[name].append(size[name][-1])
            else:
                domain = extend_by_swaps(domain, 1)
                size[name].append(len(domain))

    save_domain_size_csv(size, num_candidates)


def plot_domain_size_total(base, num_candidates) -> None:
    """Plots domain sizes from CSV data."""

    size = load_domain_size_csv(num_candidates)

    plt.figure(figsize=(4.8, 6.4))

    max_num_swaps = _get_max_num_swaps(num_candidates)
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


def compute_size_increase(size):
    """Helper function to compute size increase from size data."""
    size_increase = {}
    for name in size:
        size_increase[name] = [size[name][i] - size[name][i - 1] for i in range(1, len(size[name]))]
    return size_increase


def plot_domain_size_increase(base, num_candidates) -> None:
    """Plots domain sizes from CSV data."""

    size = load_domain_size_csv(num_candidates)

    max_num_swaps = _get_max_num_swaps(num_candidates)
    x_range = list(range(0, max_num_swaps + 1))

    # Calculate the increase in domain size
    size_increase = compute_size_increase(size)

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

        max_num_swaps = _get_max_num_swaps(num_candidates)
        x_range = list(range(0, max_num_swaps + 1))

        # Calculate the increase in domain size
        size_increase = compute_size_increase(size)

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

    candidate_range = [10]
    # candidate_range = [3]

    for num_candidates in candidate_range:
        compute_domain_size(base, num_candidates)

        # plot_domain_size_total(base, num_candidates)
        # plot_domain_size_increase(base, num_candidates)

    print_domain_diversity(base, candidate_range)


# Bla bla bla, side sentence, bla bla.
