import itertools

from src.domain.popularity import *
from src.print_utils import *
import numpy as np
import matplotlib.pyplot as plt


def plot_popularity_histogram(popularity, m, name, img_path):

    plt.figure(figsize=(8, 3))
    min_pop = 0
    max_pop = 1200
    bin_size = 50
    bins = np.arange(min_pop, max_pop + bin_size+1, bin_size)
    # Ensure cut bin is strictly greater than last bin edge
    popularity = np.array(popularity)

    y_max = len(popularity)
    plt.ylim(0 * y_max, y_max * 1)
    yticks = [0, y_max * 0.25, y_max * 0.5, y_max * 0.75, y_max]
    ytick_labels = ['0', '25%', '50%', '75%', '100%']
    plt.yticks(yticks, ytick_labels)
    plt.grid(axis='y', color='gray', alpha=0.6)  # Add pale horizontal lines

    clipped_popularity = np.where(popularity > max_pop, max_pop+1, popularity)
    print(clipped_popularity)
    counts, bins, patches = plt.hist(clipped_popularity,
                                     bins=bins,
                                     align="mid",
                                     rwidth=0.8,
                                     color='tab:blue')
    print(counts)
    # Custom bin labels for ranges
    bin_labels = [f"[{int(bins[i])},{int(bins[i+1]-1)}]" for i in range(len(bins)-2)] + [f"{max_pop}+"]
    bin_centers = bins[:-1] + (np.diff(bins) / 2)
    # Show only every 8th bin and the last bin
    tick_indices = list(range(0, len(bin_centers)-1, 8)) + [len(bin_centers)-1]
    tick_centers = [bin_centers[i] for i in tick_indices]
    tick_labels = [bin_labels[i] for i in tick_indices]
    # plt.xlabel("Popularity (number of L(C) votes closest to domain vote)", fontsize=16)
    plt.ylabel("Votes Share", fontsize=28)
    plt.title(f'{LABEL[name]}', fontsize=48)
    plt.xticks(tick_centers, tick_labels, fontsize=24, rotation=0)
    plt.yticks(fontsize=24)
    plt.xlim(min_pop, max_pop+3*bin_size/2)
    plt.tight_layout()
    plt.savefig(img_path, bbox_inches='tight', dpi=200)
    plt.show()


def compute_data_for_popularity_histogram(num_candidates, name):
    D = domains[name](num_candidates)
    LC = list(itertools.permutations(range(num_candidates)))

    popularity = compute_popularity(D, LC)
    print(popularity)
    csv_path = f'data/popularity/{name}_m{num_candidates}.csv'
    export_popularity_to_csv(popularity, D, csv_path)


def generate_popularity_histogram(num_candidates, name):
    csv_path = f'data/popularity/{name}_m{num_candidates}.csv'
    img_path = f'images/popularity/{name}_m{num_candidates}.png'
    popularity, votes = import_popularity_from_csv(csv_path)
    plot_popularity_histogram(popularity, num_candidates, name, img_path)


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
        'largest_condorcet',
    ]

    num_candidates = 8

    for name in reversed(base):
        print(f"Processing domain: {name} with {num_candidates} candidates...")

        compute_data_for_popularity_histogram(num_candidates, name)
        generate_popularity_histogram(num_candidates, name)