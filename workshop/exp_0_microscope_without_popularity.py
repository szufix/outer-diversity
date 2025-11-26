import csv
import math
import os
from itertools import permutations

import mapof.elections as mapof
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from src.domain import *
from src.domain.popularity import import_popularity_from_csv
from src.print_utils import LABEL
from image_processing import create_image_grid

def get_permutations(s):
    return [list(p) for p in permutations(range(s))]


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
    'largest_fishburn': largest_fishburn_domain,
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



def export_data_to_csv(election, saveas):
    A = election.distinct_votes
    A = [[int(x) for x in vote] for vote in A]  # convert to list of lists

    X = election.coordinates['vote'][:, 0]
    Y = election.coordinates['vote'][:, 1]

    if len(X) != len(Y):
        raise ValueError("X and Y must be the same length.")

    out_dir = os.path.join(os.getcwd(), 'data', 'microscope')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{saveas}.csv")
    with open(out_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["A", "X", "Y"])  # header
        for a, x, y in zip(A, X, Y):
            writer.writerow([a, x, y])

    print(f"Exported {len(X)} points to {out_path}")


def import_data_from_csv(filename):
    A, X, Y = [], [], []
    print(filename)
    in_path = os.path.join(os.getcwd(), 'data', 'microscope', f"{filename}.csv")
    with open(in_path, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            A.append([int(x) for x in row["A"].strip('[]').split(',')])
            X.append(float(row["X"]))
            Y.append(float(row["Y"]))

    X = np.array(X)
    Y = np.array(Y)

    print(f"Imported {len(X)} points from {in_path}")
    return A, X, Y

def _plot_x(election, centers):
    X = [election.coordinates['vote'][w][0] for w in centers]
    Y = [election.coordinates['vote'][w][1] for w in centers]
    election.microscope.ax.scatter(X, Y, color='black', s=500, marker='*')

def microscope_colored_by_candidate(
        election,
        title,
        saveas,
        candidate_id=None):

    COLORS = [
        'tab:blue',
        'tab:pink',
        'tab:orange',
        'tab:gray',
        'tab:green',
        'tab:olive',
        'tab:red',
        'tab:cyan',
        'tab:purple',
        'tab:brown',]


    ctr = 0
    for i in range(len(election.coordinates['vote'])):


        election.microscope.ax.scatter(election.coordinates['vote'][i][0],
                                       election.coordinates['vote'][i][1],
                                       alpha=0.67,
                                       s=80,
                                       color=COLORS[election.votes[i][candidate_id] % len(COLORS)]
                                       )
        ctr += 1

    election.microscope.ax.set_title(title, fontsize=44)
    election.microscope.save_to_file(saveas=saveas)

def microscope_colored_by_first_candidate(*args, **kwargs):
    return microscope_colored_by_candidate(*args, candidate_id=0, **kwargs)

def microscope_colored_by_last_candidate(*args, **kwargs):
    return microscope_colored_by_candidate(*args, candidate_id=-1, **kwargs)

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


def compute_microscope(sampler_names, candidate_range, num_ic_votes=None, with_ic=None):

    for num_candidates in candidate_range:

        for sampler_name in sampler_names:
            # Import popularity data for this domain and m

            votes = samplers[sampler_name](num_candidates)
            votes = np.array(votes)

            election_with_ic = mapof.generate_ordinal_election_from_votes(votes)
            election_with_ic.compute_distances(distance_id='swap', object_type='vote')
            election_with_ic.embed(algorithm='mds', object_type='vote')
            election_with_ic.set_microscope(alpha=.5, object_type='vote')

            if with_ic:
                saveas = f'{sampler_name}_m{num_candidates}_with_ic_{num_ic_votes}'
            else:
                saveas = f'{sampler_name}_m{num_candidates}'

            export_data_to_csv(election_with_ic, saveas)


def plot_microscope_for_different_models(base, num_candidates, min_pop=None, max_pop=None, with_ic=False,
                    num_ic_votes=None, with_title=True, title=None):
    for sampler_name in base:

        if with_ic:
            filename = f'{sampler_name}_m{num_candidates}_with_ic_{num_ic_votes}'
        else:
            filename = f'{sampler_name}_m{num_candidates}'

        votes, X, Y = import_data_from_csv(filename)
        election = mapof.generate_ordinal_election_from_votes(votes)
        election.coordinates['vote'] = np.column_stack((X, Y))

        # election.set_microscope(alpha=.0, object_type='vote') # automatic radius
        if with_title:
            title_size = 20
        else:
            title_size = None

        radius = None

        # FOR AAMAS
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

        # if title is None:
        title = f'{LABEL.get(sampler_name, sampler_name)}'

        saveas = filename

        print_microscope(election, title, saveas, min_pop, max_pop,
                         with_title=with_title)


def get_radius(has_fixed_radius, num_candidates):
    if has_fixed_radius:
        if num_candidates == 8:
            return 16
        elif num_candidates == 6:
            return 10
    else:
        return None


def get_title_size(has_title):
    if has_title:
        return 20
    else:
        return None


def get_filename(sampler_name, num_candidates, with_ic, num_ic_votes):
    if with_ic:
        return f'{sampler_name}_m{num_candidates}_with_ic_{num_ic_votes}'
    else:
        return f'{sampler_name}_m{num_candidates}'

def get_election_with_microscope(filename, num_candidates, has_fixed_radius, has_title):

    votes, X, Y = import_data_from_csv(filename)
    election = mapof.generate_ordinal_election_from_votes(votes)
    election.coordinates['vote'] = np.column_stack((X, Y))

    election.set_microscope(
        alpha=.0,
        object_type='vote',
        radius=get_radius(has_fixed_radius, num_candidates),
        title_size=get_title_size(has_title),
    )
    return election


# NEW
def plot_microscopes(
        sampler_names,
        candidate_range,
        microscope_function,
        with_ic=False,
        num_ic_votes=None,
        has_title=True,
        title=None,
        has_fixed_radius=False,
        microscope_function_params=None,
):

    for num_candidates in candidate_range:

        for sampler_name in sampler_names:

            filename = get_filename(sampler_name, num_candidates, with_ic, num_ic_votes)

            election = get_election_with_microscope(filename, num_candidates, has_fixed_radius, has_title)

            title = f'{LABEL.get(sampler_name, sampler_name)}'


            microscope_function(
                election,
                title,
                filename,
                **microscope_function_params)





# USE CASE 2
def exp_x_samplers_for_8_candidates(skip_computation=False):
    sampler_names = [
        'euclidean_3d',
        'caterpillar',
        'spoc',
        'euclidean_2d',
        'sp_double_forked',
        'balanced',
        'largest_condorcet',
        'single_peaked',
        'single_crossing',
    ]
    candidate_range = [8]

    if not skip_computation:
        compute_microscope(sampler_names, candidate_range)

    plot_microscopes(
        sampler_names=sampler_names,
        candidate_range=candidate_range,
        microscope_function=microscope_colored_by_first_candidate,
        microscope_function_params={},
    )


# exp_fishburn_for_x_candidates(skip_computation=True)
# exp_x_samplers_for_8_candidates(skip_computation=True)



##### OLD STUFF

# for num_candidates in [8]:

# num_ic_votes = 512
#
# base_sorted = [
#     'euclidean_3d',
#     'caterpillar',
#     'spoc',
#     'euclidean_2d',
#     'sp_double_forked',
#     'balanced',
#     'largest_condorcet',
#     'single_peaked',
#     'single_crossing',
# ]
#
# num_candidates = 8
#
# # WITHOUT IC
# sampler_name = 'largest_fishburn'
# candidate_range = range(2, 8)
# # compute_microscope(num_candidates, base_sorted)
# plot_microscopes(
#     sampler_names=[sampler_name],
#     candidate_range=candidate_range,
#     microscope_function=microscope_colored_by_first_candidate,
#     microscope_function_params={},
# )
# paths = [f'images/online/{name}_m{num_candidates}.png' for name in base_sorted]
# create_image_grid(paths,9,1, output_path=f'images/microscope/microscope_m{num_candidates}.png')


    # WITH IC
    # compute_microscope(num_candidates, base_sorted, num_ic_votes=num_ic_votes, with_ic=True)
    # plot_microscope(base_sorted, num_candidates, min_pop, max_pop, with_ic=True, num_ic_votes=num_ic_votes)
    # paths = [f'images/online/{name}_m{num_candidates}_with_ic_{num_ic_votes}.png' for name in base_sorted]
    # create_image_grid(paths,3,3, output_path=f'images/microscope/microscope_m{num_candidates}_with_ic.png')

# TMP
# paths = [f'images/online/largest_fishburn_m{c}.png' for c in [2,3,4,5,6,7,8,9,10]]
# create_image_grid(paths,9,1, output_path=f'images/microscope/fishburn_microscope.png')
