from src.domain.popularity import *
from src.print_utils import *

def get_global_popularity_bins(base, num_candidates):
    min_pop = float('inf')
    max_pop = float('-inf')
    for name in base:
        csv_path = f'data/popularity/{name}_m{num_candidates}.csv'
        popularity, _ = import_popularity_from_csv(csv_path)
        min_pop = min(min_pop, min(popularity))
        max_pop = max(max_pop, max(popularity))
    # Bin size = 3
    bins = list(range((min_pop // 3) * 3, ((max_pop // 3) + 2) * 3, 3))
    return bins


def plot_popularity_histogram(popularity, m, name, img_path):
    plt.figure(figsize=(8, 5))
    plt.hist(popularity,
             align="left",
             rwidth=0.8,
             color='tab:blue')
    plt.xlabel("Popularity (number of L(C) votes closest to domain vote)", fontsize=16)
    plt.ylabel("Number of Votes", fontsize=16)
    plt.title(f"Histogram of Popularity (m={m}) for {LABEL[name]}", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(img_path)


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
        # 'euclidean_3d',
        # 'euclidean_2d',
        # 'spoc',
        # 'sp_double_forked',
        # 'caterpillar',
        # 'balanced',
        # 'single_peaked',
        'single_crossing',
        # 'euclidean_1d',
    ]


    num_candidates = 6

    for name in reversed(base):
        print(f"Processing domain: {name} with {num_candidates} candidates...")

        compute_data_for_popularity_histogram(num_candidates, name)

        generate_popularity_histogram(num_candidates, name)
