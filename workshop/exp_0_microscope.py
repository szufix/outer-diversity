import csv
import os
from itertools import permutations

import mapof.elections as mapof
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from src.domain import *
from src.domain.popularity import import_popularity_from_csv
from src.print_utils import LABEL


def get_permutations(s):
    return [list(p) for p in permutations(range(s))]


# base = [
#     'euclidean_2d', 'euclidean_1d', 'single_crossing', 'single_peaked', 'sp_double_forked', 'spoc', 'caterpillar', 'balanced',
# ]

samplers = {
    'euclidean_3d': euclidean_3d_domain,
    #

    'euclidean_2d': euclidean_2d_domain,
    'euclidean_1d': euclidean_1d_domain,
    'single_crossing': single_crossing_domain,
    'single_peaked': single_peaked_domain,

    'caterpillar': group_separable_caterpillar_domain,
    'balanced': group_separable_balanced_domain,
    'spoc': spoc_domain,
    'sp_double_forked': sp_double_forked_domain,

    # 'ext_euclidean_2d': ext_euclidean_2d_domain,
    # 'ext_euclidean_1d': ext_euclidean_1d_domain,
    # 'ext_single_peaked': ext_single_peaked_domain,
    # 'ext_single_crossing': ext_single_crossing_domain,

    # 'sc_with_gaps': single_crossing_with_gaps,
    # 'weighted_sc_with_gaps': weighted_single_crossing_with_gaps,
    'largest_condorcet': largest_condorcet_domain,
    'fishburn': fishburn_domain,
}

COLORS = ['black',
          'tab:blue',
          'tab:orange',
          'tab:green',
          'tab:red',
          'tab:purple',
          'tab:brown',
          'tab:pink',
          'tab:gray',
          'tab:olive',
          'tab:cyan']



def export_data_to_csv(election, popularity, saveas):
    A = election.distinct_votes
    A = [[int(x) for x in vote] for vote in A]  # convert to list of lists

    X = election.coordinates['vote'][:, 0]
    Y = election.coordinates['vote'][:, 1]

    if len(X) != len(Y):
        raise ValueError("X and Y must be the same length.")

    print('pop', popularity)

    out_dir = os.path.join(os.getcwd(), 'data', 'microscope')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{saveas}.csv")
    with open(out_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["A", "X", "Y", "popularity"])  # header
        for a, x, y, p in zip(A, X, Y, popularity):
            writer.writerow([a, x, y, p])

    print(f"Exported {len(X)} points to {out_path}")


def import_data_from_csv(filename):
    A, X, Y, popularity = [], [], [], []
    print(filename)
    in_path = os.path.join(os.getcwd(), 'data', 'microscope', f"{filename}.csv")
    with open(in_path, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            A.append([int(x) for x in row["A"].strip('[]').split(',')])
            X.append(float(row["X"]))
            Y.append(float(row["Y"]))
            popularity.append(float(row["popularity"]))

    X = np.array(X)
    Y = np.array(Y)
    popularity = np.array(popularity)

    print(f"Imported {len(X)} points from {in_path}")
    return A, X, Y, popularity

def _plot_x(election, centers):
    X = [election.coordinates['vote'][w][0] for w in centers]
    Y = [election.coordinates['vote'][w][1] for w in centers]
    election.microscope.ax.scatter(X, Y, color='black', s=500, marker='*')

def print_microscope(
        election,
        popularity,
        title,
        saveas,
        min_pop=None,
        max_pop=None,
        with_title=True):

    # Set normalization limits and center
    ideal_pop = 40320

    if min_pop is None:
        min_pop = 1
    if max_pop is None:
        max_pop = 200000

    popularity_without_ic = np.array([p for p in popularity if p >= 0])
    popularity_without_ic *= len(popularity_without_ic)

    print('min_pop', min_pop)
    print('max_pop', max_pop)
    clipped_popularity_without_ic = np.clip(popularity_without_ic, min_pop, max_pop)

    colors_list = ["#08519c", "#f5deb3", "#d73027"]
    custom_cmap = LinearSegmentedColormap.from_list('blue_green_red', colors_list, N=256)
    norm = TwoSlopeNorm(vmin=min_pop, vcenter=ideal_pop, vmax=max_pop)
    colors = custom_cmap(norm(clipped_popularity_without_ic))

    for i in range(len(popularity)):
        if popularity[i] == -1:
            election.microscope.ax.scatter(election.coordinates['vote'][i][0],
                                           election.coordinates['vote'][i][1],
                                           c='#f0f0f0',
                                           # alpha=0.05,
                                           s=25)
    ctr = 0
    for i in range(len(popularity)):
        if popularity[i] >= 0:

            if popularity_without_ic[ctr] > ideal_pop:
                marker = 'o'  # upward triangle for above ideal
                edge = None
                lw = None
            elif popularity_without_ic[ctr] == ideal_pop:
                marker = 'o'  # circle with thin black border for ideal
                edge = 'black'
                lw = 1
            elif popularity_without_ic[ctr] < ideal_pop:
                marker = 'x'  # downward triangle for below ideal
                edge = None
                lw = None


            election.microscope.ax.scatter(election.coordinates['vote'][i][0],
                                           election.coordinates['vote'][i][1],
                                           c=[colors[ctr]],
                                           alpha=1,
                                           marker=marker,
                                           s=80,
                                           edgecolors=edge,
                                           linewidths=lw)
            ctr += 1

        election.microscope.ax.set_title(title, fontsize=44)
    # election.microscope.show_and_save(saveas=saveas)
    election.microscope.save_to_file(saveas=saveas)

def map_ids(election_target, election_with_ic, centers, clusters):
    new_center_ids = []
    new_cluster_ids = []
    for i, vote in enumerate(election_with_ic.distinct_votes):

        if tuple(vote) in centers:
            new_center_ids.append(i)

        if vote in election_target.distinct_votes:
            new_cluster_ids.append(clusters[tuple(vote)])
        else:
            new_cluster_ids.append(0)

    return new_center_ids, new_cluster_ids


def compute_microscope(num_candidates, base, num_ic_votes=None, with_ic=None):

    for sampler_name in base:
        # Import popularity data for this domain and m
        pop_csv_path = os.path.join(os.getcwd(), f'data/popularity/{sampler_name}_m{num_candidates}.csv')
        popularity, pop_votes = import_popularity_from_csv(pop_csv_path)

        if with_ic:
            votes_with_ic = [np.random.permutation(num_candidates) for _ in range(num_ic_votes)]
        else:
            votes_with_ic = []

        votes = pop_votes + votes_with_ic
        votes = np.array(votes)

        election_with_ic = mapof.generate_ordinal_election_from_votes(votes)
        election_with_ic.compute_distances(distance_id='swap', object_type='vote')
        election_with_ic.embed(algorithm='mds', object_type='vote')
        election_with_ic.set_microscope(alpha=.5, object_type='vote')

        if with_ic:
            saveas = f'{sampler_name}_m{num_candidates}_with_ic_{num_ic_votes}'
        else:
            saveas = f'{sampler_name}_m{num_candidates}'

        # Map votes to popularity properly
        vote_to_pop = {tuple(v): p for v, p in zip(pop_votes, popularity)}
        mapped_popularity = [vote_to_pop.get(tuple(v), -1.) for v in election_with_ic.distinct_votes]

        export_data_to_csv(election_with_ic, mapped_popularity, saveas)


def plot_microscope(base, num_candidates, min_pop, max_pop, with_ic=False, num_ic_votes=None, with_title=True):
    for sampler_name in base:

        if with_ic:
            filename = f'{sampler_name}_m{num_candidates}_with_ic_{num_ic_votes}'
        else:
            filename = f'{sampler_name}_m{num_candidates}'

        votes, X, Y, popularity = import_data_from_csv(filename)
        election = mapof.generate_ordinal_election_from_votes(votes)
        election.coordinates['vote'] = np.column_stack((X, Y))

        # election.set_microscope(alpha=.0, object_type='vote') # automatic radius
        if with_title:
            title_size = 20
        else:
            title_size = None

        radius = None

        if num_candidates == 8:
            radius = 16
        elif num_candidates == 6:
            radius = 10


        if radius is None:
            election.set_microscope(
                alpha=.0,
                object_type='vote',
                title_size=title_size
            )
        else:
            election.set_microscope(
                alpha=.0,
                object_type='vote',
                radius=radius, # fixed radius
                title_size=title_size
            )

        title = f'{LABEL.get(sampler_name, sampler_name)}'

        saveas = filename

        print_microscope(election, popularity, title, saveas, min_pop, max_pop, with_title=with_title)

def save_microscope_colorbar(min_pop=1, max_pop=200000, ideal_pop=40320, filename='microscope_colorbar.png'):


    colors_list = ["#08519c", "#f5deb3", "#d73027"]
    custom_cmap = LinearSegmentedColormap.from_list('blue_green_red', colors_list, N=256)
    norm = TwoSlopeNorm(vmin=min_pop, vcenter=ideal_pop, vmax=max_pop)

    fig, ax = plt.subplots(figsize=(1, 3))
    fig.subplots_adjust(left=0.3, right=0.7)
    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=custom_cmap),
        cax=ax,
        orientation='vertical',
        label='Popularity',
    )
    cb.set_label('N. Popularity', fontsize=30)
    cb.ax.tick_params(labelsize=32)

    # Only show ticks for min, optimal, and max values
    cb.set_ticks([min_pop, ideal_pop, max_pop])

    n_min = int(min_pop/ideal_pop)
    n_ideal = 1
    n_max = int(max_pop/ideal_pop)
    cb.set_ticklabels([str(n_min), str(n_ideal), str(n_max) + '+'])


    fig.savefig(filename, bbox_inches='tight', dpi=200)
    plt.show()

# To generate the colorbar PNG, call:
save_microscope_colorbar()

num_candidates = 8
num_ic_votes = 512

min_pop = 1
max_pop = 200000


base_sorted = [
    # 'euclidean_3d',
    # 'caterpillar',
    # 'spoc',
    # 'euclidean_2d',
    # 'sp_double_forked',
    # 'balanced',
    'largest_condorcet',
    # 'single_peaked',
    # 'single_crossing',
]


# WITHOUT IC
# compute_microscope(num_candidates, base_sorted)
# plot_microscope(base_sorted, num_candidates, min_pop, max_pop)
# paths = [f'images/online/{name}_m{num_candidates}.png' for name in base_sorted]
# create_image_grid(paths,9,1, output_path=f'images/microscope/microscope_m{num_candidates}.png')


# WITH IC
# compute_microscope(num_candidates, base_sorted, num_ic_votes=num_ic_votes, with_ic=True)
# plot_microscope(base_sorted, num_candidates, min_pop, max_pop, with_ic=True, num_ic_votes=num_ic_votes)
# paths = [f'images/online/{name}_m{num_candidates}_with_ic_{num_ic_votes}.png' for name in base_sorted]
# create_image_grid(paths,3,3, output_path=f'images/microscope/microscope_m{num_candidates}_with_ic.png')
