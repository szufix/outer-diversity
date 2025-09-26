import matplotlib.pyplot as plt
from src.print_utils import *
from src.diversity.diversity_utils import *
from src.diversity.growing_domains import *


def print_domain_diversity(base, candidate_range, num_runs=10):
    import numpy as np
    # Store all diversity values for all runs
    diversity_data = {num_candidates: {name: [] for name in base} for num_candidates in candidate_range}

    for num_candidates in candidate_range:
        for run_idx in range(num_runs):
            try:
                size = load_domain_size_csv(f"{num_candidates}_run{run_idx}")
            except Exception:
                size = load_domain_size_csv(num_candidates)  # fallback for single file
            for name in base:
                if name not in size:
                    print(f"Warning: {name} not found in the CSV data for run {run_idx}.")
                    continue
                size_increase = compute_balls_increase(size[name])
                total_diversity = outer_diversity_from_balls_increase(size_increase, num_candidates)
                diversity_data[num_candidates][name].append(total_diversity)

    # Compute mean and std
    diversity_stats = {num_candidates: {} for num_candidates in candidate_range}
    for num_candidates in candidate_range:
        for name in base:
            vals = diversity_data[num_candidates][name]
            if len(vals) > 0:
                avg = float(np.mean(vals))
                std = float(np.std(vals))
            else:
                avg = std = 0.0
            diversity_stats[num_candidates][name] = (avg, std)

    # Sort base by avg diversity for first candidate
    first_candidate = candidate_range[0]
    sorted_base = sorted(base, key=lambda name: diversity_stats[first_candidate][name][0])

    # Print LaTeX table
    print("\\begin{tabular}{c" + "c" * len(candidate_range) + "}")
    print("\\hline")
    header = "Domain"
    for num_candidates in candidate_range:
        header += f" & {num_candidates}"
    header += " \\\\"
    print(header)
    print("\\hline")
    for name in sorted_base:
        row = LABEL[name]
        for num_candidates in candidate_range:
            avg, std = diversity_stats[num_candidates][name]
            row += f" & {round(avg, 3)} $\\pm$ {round(std, 3)}"
        row += " \\\\"
        print(row)
    print("\\hline")
    print("\\end{tabular}")



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

    plt.savefig(f'images/domain_size_m{num_candidates}_total.png', dpi=200, bbox_inches='tight')
    plt.close()


def plot_domain_size_increase(base, num_candidates) -> None:
    """Plots domain sizes from CSV data."""

    size = load_domain_size_csv(num_candidates)

    max_num_swaps = get_max_num_swaps(num_candidates)
    x_range = list(range(0, max_num_swaps + 1))

    plt.figure(figsize=(6.4, 4.8))

    for name in base:
        if name in size:
            size_increase = compute_balls_increase(size[name])
            plt.plot(x_range[1:],
                     size_increase,
                     label=LABEL[name],
                     marker=MARKER[name],
                     color=COLOR[name])
        else:
            print(f"Warning: {name} not found in the CSV data.")

    plt.xticks(x_range[1:], fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Number of swaps', fontsize=22)
    plt.ylabel('Size increase', fontsize=22)
    plt.title(f'DS Increase for {num_candidates} candidates', fontsize=22)
    plt.legend(fontsize=18, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig(f'images/domain_size_m{num_candidates}_increase.png', dpi=200, bbox_inches='tight')
    plt.close()


def plot_domain_size_increase_bar(name, num_candidates) -> None:
    """Plots domain sizes from CSV data."""

    size = load_domain_size_csv(num_candidates)

    max_num_swaps = get_max_num_swaps(num_candidates)
    x_range = list(range(0, max_num_swaps + 1))

    plt.figure(figsize=(8, 3))

    if name in size:
        size_increase = compute_balls_increase(size[name])
        plt.bar(x_range[1:],
                 size_increase,
                 label=LABEL[name],
                 color=COLOR[name])
    else:
        print(f"Warning: {name} not found in the CSV data.")

    # plt.xticks(x_range[1:], fontsize=16)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlabel('Number of swaps', fontsize=28)
    plt.ylabel('Size increase', fontsize=28)
    plt.title(f'{LABEL[name]}', fontsize=48)
    # plt.legend(fontsize=18, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.ylim([0, 7500])
    plt.savefig(f'images/domain_size/domain_size_{name}_m{num_candidates}_bar.png',
                dpi=200, bbox_inches='tight')
    # plt.show()
    plt.close()



if __name__ == "__main__":

    base = [
        'euclidean_3d',
        'euclidean_2d',
        'spoc',
        'sp_double_forked',
        'caterpillar',
        'balanced',
        'single_peaked',
        'single_crossing',
        'euclidean_1d',
        'ext_single_vote',
        'single_vote',
    ]

    candidate_range = [6,8]
    num_runs = 10

    # for num_candidates in candidate_range:
    #     compute_domain_balls(base, num_candidates, num_runs)

        # plot_domain_size_total(base, num_candidates)
        # plot_domain_size_increase(base, num_candidates)

        # for name in base:
        #     plot_domain_size_increase_bar(name, num_candidates)

    print_domain_diversity(base, candidate_range, num_runs)
