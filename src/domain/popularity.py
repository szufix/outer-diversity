import itertools
import matplotlib.pyplot as plt
import csv
from src.diversity.diversity_utils import swap_distance_between_potes, vote_to_pote

from src.domain import *

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

def compute_popularity(domain_votes, lc_votes):
    D_potes = [vote_to_pote(v) for v in domain_votes]
    LC_potes = [vote_to_pote(list(v)) for v in lc_votes]
    popularity = [0 for _ in domain_votes]
    for lc_pote in LC_potes:
        min_dist = None
        min_idxs = []
        for i, d_pote in enumerate(D_potes):
            dist = swap_distance_between_potes(lc_pote, d_pote)
            if min_dist is None or dist < min_dist:
                min_dist = dist
                min_idxs = [i]
            elif dist == min_dist:
                min_idxs.append(i)
        for idx in min_idxs:
            popularity[idx] += 1. / len(min_idxs)
    return popularity


def export_popularity_to_csv(popularity, votes, path):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['popularity', 'votes'])  # header
        for pop, v in zip(popularity, votes):
            writer.writerow([pop, v])


def import_popularity_from_csv(path):
    popularity = []
    votes = []
    print(path)
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            popularity.append(float(row['popularity']))
            votes.append([int(x) for x in row["votes"].strip('[]').split(',')])

    return popularity, votes


