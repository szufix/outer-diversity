import glob
from time import time
import csv
import os

from src.print_utils import *
import pandas as pd

from src.diversity.sampling import (
    outer_diversity_sampling,
    outer_diversity_sampling_for_structered_domains
)
from src.domain.single_peaked import single_peaked_domain
from src.domain.group_separable import (
    group_separable_caterpillar_domain,
    group_separable_balanced_domain
)
from src.max_diversity.main import find_optimal_facilities_sampled_simulated_annealing
import multiprocessing
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_joint_diversity_comparison(candidate_range):
    """
    Plot diversity comparison from joint CSV, showing mean and std across all runs.
    Also print the mean and std for each candidate count.
    """

    max_candidate = candidate_range[-1]
    # max_candidate -=

    results_dir = os.path.join(os.path.dirname(__file__), 'data', 'changing_m')
    csv_path = os.path.join(results_dir, '_single_peaked_joint_no_max.csv')
    df = pd.read_csv(csv_path)
    grouped = df.groupby('num_candidates')
    candidate_range = np.array(sorted(df['num_candidates'].unique()))
    sp_mean = grouped['sp_diversity'].mean().values
    sp_std = grouped['sp_diversity'].std().values
    gs_caterpillar_mean = grouped['gs_caterpillar_diversity'].mean().values
    gs_caterpillar_std = grouped['gs_caterpillar_diversity'].std().values
    gs_balanced_mean = grouped['gs_balanced_diversity'].mean().values
    gs_balanced_std = grouped['gs_balanced_diversity'].std().values

    # Read single crossing results
    csv_path_sc = os.path.join(results_dir, '_single_crossing_joint.csv')
    df_sc = pd.read_csv(csv_path_sc)
    grouped_sc = df_sc.groupby('num_candidates')
    sc_mean = grouped_sc['sc_diversity'].mean().values
    sc_std = grouped_sc['sc_diversity'].std().values
    sc_candidate_range = np.array(sorted(df_sc['num_candidates'].unique()))

    # Read euclidean 2d results
    csv_path_2d = os.path.join(results_dir, '_euclidean_2d_joint_no_max.csv')
    df_2d = pd.read_csv(csv_path_2d)
    grouped_2d = df_2d.groupby('num_candidates')
    twod_mean = grouped_2d['2d_diversity'].mean().values
    twod_std = grouped_2d['2d_diversity'].std().values
    twod_candidate_range = np.array(sorted(df_2d['num_candidates'].unique()))


    # Read spoc results
    csv_path_spoc = os.path.join(results_dir, '_spoc_joint.csv')
    df_spoc = pd.read_csv(csv_path_spoc)
    grouped_spoc = df_spoc.groupby('num_candidates')
    spoc_mean = grouped_spoc['spoc_diversity'].mean().values
    spoc_std = grouped_spoc['spoc_diversity'].std().values
    spoc_candidate_range = np.array(sorted(df_spoc['num_candidates'].unique()))

    # Read euclidean 3d results
    csv_path_3d = os.path.join(results_dir, '_euclidean_3d_joint.csv')
    df_3d = pd.read_csv(csv_path_3d)
    grouped_3d = df_3d.groupby('num_candidates')
    threed_mean = grouped_3d['euclidean_3d_diversity'].mean().values
    threed_std = grouped_3d['euclidean_3d_diversity'].std().values
    threed_candidate_range = np.array(sorted(df_3d['num_candidates'].unique()))

    gs_balanced_mean = gs_balanced_mean[0:max_candidate]
    gs_balanced_std = gs_balanced_std[0:max_candidate]
    gs_caterpillar_mean = gs_caterpillar_mean[0:max_candidate]
    gs_caterpillar_std = gs_caterpillar_std[0:max_candidate]
    sp_mean = sp_mean[0:max_candidate]
    sp_std = sp_std[0:max_candidate]
    sc_mean = sc_mean[0:max_candidate]
    sc_std = sc_std[0:max_candidate]
    spoc_mean = spoc_mean[0:max_candidate]
    spoc_std = spoc_std[0:max_candidate]
    threed_mean = threed_mean[0:max_candidate]
    threed_std = threed_std[0:max_candidate]
    twod_mean = twod_mean[0:max_candidate]
    twod_std = twod_std[0:max_candidate]

    print(len(candidate_range))
    print(len(gs_balanced_mean))
    print(len(threed_mean))


    plt.figure(figsize=(8, 12))

    plt.plot(candidate_range, gs_caterpillar_mean,
             label=LABEL['caterpillar'],
             marker=MARKER['caterpillar'], linewidth=2, markersize=8, color=COLOR['caterpillar'])
    plt.fill_between(candidate_range, gs_caterpillar_mean - gs_caterpillar_std,
                     gs_caterpillar_mean + gs_caterpillar_std, color=COLOR['caterpillar'],
                     alpha=0.2)

    plt.plot(candidate_range, threed_mean, label=LABEL['euclidean_3d'],
             marker=MARKER['euclidean_3d'], linewidth=2, markersize=8, color=COLOR['euclidean_3d'])
    plt.fill_between(candidate_range, threed_mean - threed_std, threed_mean + threed_std,
                     color=COLOR['euclidean_3d'], alpha=0.2)

    plt.plot(candidate_range, spoc_mean, label=LABEL['spoc'],
             marker=MARKER['spoc'], linewidth=2, markersize=8, color=COLOR['spoc'])
    plt.fill_between(candidate_range, spoc_mean - spoc_std, spoc_mean + spoc_std,
                     color=COLOR['spoc'], alpha=0.2)

    plt.plot(candidate_range, twod_mean, label='2D-Sqr.',
             marker=MARKER['euclidean_2d'], linewidth=2, markersize=8, color=COLOR['euclidean_2d'])
    plt.fill_between(candidate_range, twod_mean - twod_std, twod_mean + twod_std,
                     color=COLOR['euclidean_2d'], alpha=0.2)

    plt.plot(candidate_range, gs_balanced_mean, label=LABEL['balanced'], marker=MARKER['balanced'],
             linewidth=2, markersize=8, color=COLOR['balanced'])
    plt.fill_between(candidate_range, gs_balanced_mean - gs_balanced_std,
                     gs_balanced_mean + gs_balanced_std, color=COLOR['balanced'], alpha=0.2)

    plt.plot(candidate_range, sp_mean, label=LABEL['single_peaked'], marker=MARKER['single_peaked'],
             linewidth=2, markersize=8, color=COLOR['single_peaked'])
    plt.fill_between(candidate_range, sp_mean - sp_std, sp_mean + sp_std,
                     color=COLOR['single_peaked'], alpha=0.2)

    plt.plot(candidate_range, sc_mean, label=LABEL['single_crossing'],
             marker=MARKER['single_crossing'], linewidth=2, markersize=8,
             color=COLOR['single_crossing'])
    plt.fill_between(candidate_range, sc_mean - sc_std, sc_mean + sc_std,
                     color=COLOR['single_crossing'], alpha=0.2)


    plt.xlabel('Number of Candidates', fontsize=36)
    plt.ylabel('Outer Diversity', fontsize=36)
    plt.legend(fontsize=24, loc='lower left')
    plt.grid(True, alpha=0.3)
    xticks_to_show = [2, 5, 8, 11, 14, 17]
    plt.xticks(xticks_to_show, fontsize=28)
    plt.yticks(fontsize=28)
    plt.ylim([-0.05, 1.05])
    plt.tight_layout()
    plt.savefig('images/changing_m/changing_m_all_domains_with_max.png', dpi=300,
                bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    candidate_range = range(2,17)
    num_samples = 1000
    max_iterations = 256
    num_runs = 1
    plot_joint_diversity_comparison(candidate_range)
