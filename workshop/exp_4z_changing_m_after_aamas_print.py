import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.print_utils import *


def plot_joint_diversity_comparison(list_of_cultures, candidate_range):
    """
    Plot diversity comparison from joint CSV, showing mean and std across all runs.
    Also print the mean and std for each candidate count.
    """

    max_candidate = candidate_range[-1]

    results_dir = os.path.join(os.path.dirname(__file__), 'data', 'changing_m')

    plt.figure(figsize=(8, 12))


    if 'single_peaked' in list_of_cultures:
        csv_path = os.path.join(results_dir, '_single_peaked_joint.csv')
        df = pd.read_csv(csv_path)
        grouped = df.groupby('num_candidates')
        candidate_range = np.array(sorted(df['num_candidates'].unique()))
        sp_mean = grouped['single_peaked_diversity'].mean().values
        sp_std = grouped['single_peaked_diversity'].std().values

        sp_mean = sp_mean[0:max_candidate]
        sp_std = sp_std[0:max_candidate]

    if 'caterpillar' in list_of_cultures:
        csv_path = os.path.join(results_dir, '_caterpillar_joint.csv')
        df = pd.read_csv(csv_path)
        grouped = df.groupby('num_candidates')
        gs_caterpillar_mean = grouped['caterpillar_diversity'].mean().values
        gs_caterpillar_std = grouped['caterpillar_diversity'].std().values
        gs_caterpillar_mean = gs_caterpillar_mean[0:max_candidate]
        gs_caterpillar_std = gs_caterpillar_std[0:max_candidate]

    if 'balanced' in list_of_cultures:
        csv_path = os.path.join(results_dir, '_balanced_joint.csv')
        df = pd.read_csv(csv_path)
        grouped = df.groupby('num_candidates')
        gs_balanced_mean = grouped['balanced_diversity'].mean().values
        gs_balanced_std = grouped['balanced_diversity'].std().values
        gs_balanced_mean = gs_balanced_mean[0:max_candidate]
        gs_balanced_std = gs_balanced_std[0:max_candidate]

    # Read single crossing results
    if 'single_crossing' in list_of_cultures:
        csv_path_sc = os.path.join(results_dir, '_single_crossing_joint.csv')
        df_sc = pd.read_csv(csv_path_sc)
        grouped_sc = df_sc.groupby('num_candidates')
        sc_mean = grouped_sc['single_crossing_diversity'].mean().values
        sc_std = grouped_sc['single_crossing_diversity'].std().values
        sc_candidate_range = np.array(sorted(df_sc['num_candidates'].unique()))

        sc_mean = sc_mean[0:max_candidate]
        sc_std = sc_std[0:max_candidate]

    if 'euclidean_2d' in list_of_cultures:
        csv_path_2d = os.path.join(results_dir, '_euclidean_2d_joint.csv')
        df_2d = pd.read_csv(csv_path_2d)
        grouped_2d = df_2d.groupby('num_candidates')
        twod_mean = grouped_2d['euclidean_2d_diversity'].mean().values
        twod_std = grouped_2d['euclidean_2d_diversity'].std().values
        twod_candidate_range = np.array(sorted(df_2d['num_candidates'].unique()))

        twod_mean = twod_mean[0:max_candidate]
        twod_std = twod_std[0:max_candidate]

    if 'spoc' in list_of_cultures:
        csv_path_spoc = os.path.join(results_dir, '_spoc_joint.csv')
        df_spoc = pd.read_csv(csv_path_spoc)
        grouped_spoc = df_spoc.groupby('num_candidates')
        spoc_mean = grouped_spoc['spoc_diversity'].mean().values
        spoc_std = grouped_spoc['spoc_diversity'].std().values
        spoc_candidate_range = np.array(sorted(df_spoc['num_candidates'].unique()))

        spoc_mean = spoc_mean[0:max_candidate]
        spoc_std = spoc_std[0:max_candidate]

    if 'euclidean_3d' in list_of_cultures:
        csv_path_3d = os.path.join(results_dir, '_euclidean_3d_joint.csv')
        df_3d = pd.read_csv(csv_path_3d)
        grouped_3d = df_3d.groupby('num_candidates')
        threed_mean = grouped_3d['euclidean_3d_diversity'].mean().values
        threed_std = grouped_3d['euclidean_3d_diversity'].std().values
        threed_candidate_range = np.array(sorted(df_3d['num_candidates'].unique()))

        threed_mean = threed_mean[0:max_candidate]
        threed_std = threed_std[0:max_candidate]


    if 'caterpillar' in list_of_cultures:
        plt.plot(candidate_range, gs_caterpillar_mean,
                 label=LABEL['caterpillar'],
                 marker=MARKER['caterpillar'], linewidth=2, markersize=8, color=COLOR['caterpillar'])
        plt.fill_between(candidate_range, gs_caterpillar_mean - gs_caterpillar_std,
                         gs_caterpillar_mean + gs_caterpillar_std, color=COLOR['caterpillar'],
                         alpha=0.2)

    if 'euclidean_3d' in list_of_cultures:
        plt.plot(threed_candidate_range, threed_mean, label=LABEL['euclidean_3d'],
                 marker=MARKER['euclidean_3d'], linewidth=2, markersize=8, color=COLOR['euclidean_3d'])
        plt.fill_between(threed_candidate_range, threed_mean - threed_std, threed_mean + threed_std,
                         color=COLOR['euclidean_3d'], alpha=0.2)

    if 'spoc' in list_of_cultures:
        plt.plot(spoc_candidate_range, spoc_mean, label=LABEL['spoc'],
                 marker=MARKER['spoc'], linewidth=2, markersize=8, color=COLOR['spoc'])
        plt.fill_between(spoc_candidate_range, spoc_mean - spoc_std, spoc_mean + spoc_std,
                         color=COLOR['spoc'], alpha=0.2)

    if 'euclidean_2d' in list_of_cultures:
        plt.plot(twod_candidate_range, twod_mean, label=SHORT_LABEL['euclidean_2d'],
                 marker=MARKER['euclidean_2d'], linewidth=2, markersize=8, color=COLOR['euclidean_2d'])
        plt.fill_between(twod_candidate_range, twod_mean - twod_std, twod_mean + twod_std,
                         color=COLOR['euclidean_2d'], alpha=0.2)

    if 'balanced' in list_of_cultures:
        plt.plot(candidate_range, gs_balanced_mean, label=LABEL['balanced'], marker=MARKER['balanced'],
                 linewidth=2, markersize=8, color=COLOR['balanced'])
        plt.fill_between(candidate_range, gs_balanced_mean - gs_balanced_std,
                         gs_balanced_mean + gs_balanced_std, color=COLOR['balanced'], alpha=0.2)

    if 'single_peaked' in list_of_cultures:
        plt.plot(candidate_range, sp_mean, label=LABEL['single_peaked'], marker=MARKER['single_peaked'],
                 linewidth=2, markersize=8, color=COLOR['single_peaked'])
        plt.fill_between(candidate_range, sp_mean - sp_std, sp_mean + sp_std,
                         color=COLOR['single_peaked'], alpha=0.2)

    if 'single_crossing' in list_of_cultures:
        plt.plot(sc_candidate_range, sc_mean, label=LABEL['single_crossing'],
                 marker=MARKER['single_crossing'], linewidth=2, markersize=8,
                 color=COLOR['single_crossing'])
        plt.fill_between(sc_candidate_range, sc_mean - sc_std, sc_mean + sc_std,
                         color=COLOR['single_crossing'], alpha=0.2)


    plt.xlabel('Number of Candidates', fontsize=36)
    plt.ylabel('Outer Diversity', fontsize=36)
    plt.legend(fontsize=27, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.axvline(x=8, color='gray', linestyle='--', linewidth=3, alpha=0.7)
    xticks_to_show = [2, 5, 8, 11, 14, 17, 20]
    plt.xticks(xticks_to_show, fontsize=28)
    plt.yticks(fontsize=28)
    plt.ylim([-0.01, 1.01])
    plt.tight_layout()
    plt.savefig('images/changing_m/changing_m_all_domains.png', dpi=300,
                bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    list_of_cultures = [
        'single_peaked',
        'single_crossing',
        'euclidean_2d',
        'euclidean_3d',
        'spoc',
        'balanced',
        'caterpillar'
    ]

    candidate_range = range(2,8)
    plot_joint_diversity_comparison(list_of_cultures, candidate_range)
