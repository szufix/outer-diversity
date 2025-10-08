import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.diversity.diversity_utils import *
from src.diversity.growing_domains import *
from src.print_utils import *


def print_domain_diversity(base, candidate_range, num_runs=10):
    import numpy as np
    # Store all diversity values for all runs
    diversity_data = {num_candidates: {name: [] for name in base} for num_candidates in candidate_range}
    dist_1_data = {num_candidates: {name: [] for name in base} for num_candidates in candidate_range}
    sizes_data = {num_candidates: {name: [] for name in base} for num_candidates in candidate_range}
    ansd_data = {num_candidates: {name: [] for name in base} for num_candidates in candidate_range}
    dist1_per_size_data = {num_candidates: {name: [] for name in base} for num_candidates in candidate_range}

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
                dist_1_data[num_candidates][name].append(size_increase[0]) # dist-1 is first increase
                sizes_data[num_candidates][name].append(size[name][0]) # initial size
                ansd_data[num_candidates][name].append((1 - total_diversity) / 2)
                dist1_per_size_data[num_candidates][name].append(size_increase[0] / size[name][0]) # dist-1 / size

    # Compute mean and std
    diversity_stats = {num_candidates: {} for num_candidates in candidate_range}
    dist_1_stats = {num_candidates: {} for num_candidates in candidate_range}
    sizes_stats = {num_candidates: {} for num_candidates in candidate_range}
    ansd_stats = {num_candidates: {} for num_candidates in candidate_range}
    dist1_per_size_stats = {num_candidates: {} for num_candidates in candidate_range}

    for num_candidates in candidate_range:
        for name in base:
            vals = diversity_data[num_candidates][name]
            avg = float(np.mean(vals))
            std = float(np.std(vals))
            diversity_stats[num_candidates][name] = (avg, std)


            vals = dist_1_data[num_candidates][name]
            avg = float(np.mean(vals))
            std = float(np.std(vals))
            dist_1_stats[num_candidates][name] = (avg, std)


            vals = sizes_data[num_candidates][name]
            avg = float(np.mean(vals))
            std = float(np.std(vals))
            sizes_stats[num_candidates][name] = (avg, std)

            vals = ansd_data[num_candidates][name]
            avg = float(np.mean(vals))
            std = float(np.std(vals))
            ansd_stats[num_candidates][name] = (avg, std)

            vals = dist1_per_size_data[num_candidates][name]
            avg = float(np.mean(vals))
            std = float(np.std(vals))
            dist1_per_size_stats[num_candidates][name] = (avg, std)


    # Sort base by avg diversity for first candidate
    first_candidate = candidate_range[0]
    sorted_base = sorted(base, key=lambda name: diversity_stats[first_candidate][name][0])


    # Print LaTeX table
    print("\\begin{tabular}{cccccc}")
    print("\\toprule")
    print("Domain $D$ & $|D|$ & $\\ansd(D)$ &$\\out(D)$ & dist-1 & dist-1/$|D|$ \\\\")
    print("\\midrule")

    for name in base:
        for num_candidates in candidate_range:
            domain_label = LABEL.get(name, name)

            avg_size, std_size = sizes_stats[num_candidates][name]
            avg_dist_1, std_dist_1 = dist_1_stats[num_candidates][name]
            avg_ansd, std_ansd = ansd_stats[num_candidates][name]
            avg_dist1_per_size, std_dist1_per_size  = dist1_per_size_stats[num_candidates][name]
            avg_diversity, std_diversity = diversity_stats[num_candidates][name]

            print(f"  {domain_label} & {int(avg_size)} & {round(avg_ansd,3)} & {round(avg_diversity,3)} & {avg_dist_1} & {round(avg_dist1_per_size,3)} \\\\")
            print(f"  {domain_label} & {int(std_size)} & {round(std_ansd,3)} & {round(std_diversity,3)} & {round(std_dist_1,3)} & {round(std_dist1_per_size,3)} \\\\")
            if name in ['ext_single_vote', 'balanced', 'spoc', 'single_crossing', 'euclidean_3d']:
                print("\\midrule")
    print("\\bottomrule")
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
    # Format y-axis ticks as 1k, 2k, ...
    ax = plt.gca()
    ax.yaxis.set_major_locator(mticker.MultipleLocator(1000))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x/1000)}k' if x >= 1000 else f'{int(x)}'))
    plt.savefig(f'images/domain_size/domain_size_{name}_m{num_candidates}_bar.png',
                dpi=200, bbox_inches='tight')
    # plt.show()
    plt.close()




if __name__ == "__main__":

    base = [
        'ext_single_vote',
        'caterpillar',
        'balanced',
        'single_peaked',
        'sp_double_forked',
        'spoc',
        'single_crossing',
        'euclidean_1d',
        'euclidean_2d',
        'euclidean_3d',
        'largest_condorcet'
    ]

    num_runs = 10
    # candidate_range = [10, 11, 12]
    # candidate_range = [2,4,6,8,10,12,14,16,18,20]
    candidate_range = [8]

    # for num_candidates in candidate_range:
        # print(num_candidates)
        # compute_domain_balls(base, num_candidates, num_runs)

        # plot_domain_size_total(base, num_candidates)
        # plot_domain_size_increase(base, num_candidates)

        # for name in base:
        #     plot_domain_size_increase_bar(name, num_candidates)

    print_domain_diversity(base, candidate_range, num_runs)
