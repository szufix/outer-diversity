import random
import numpy as np
import itertools
from tqdm import tqdm

def swap_distance_between_potes(pote_1: list, pote_2: list, m : int) -> int:
    """ Return: Swap distance between two potes """
    swap_distance = 0
    for a in range(m):
        for b in range(m):
            if (pote_1[a] < pote_1[b] and pote_2[a] >= pote_2[b]):
                swap_distance += 0.5
            if (pote_1[a] <= pote_1[b] and pote_2[a] > pote_2[b]):
                swap_distance += 0.5
    return swap_distance

def get_vote_dists(election):
    try:
        return election.vote_dists
    except:
        potes = election.get_potes()
        distances = np.zeros([election.num_voters, election.num_voters])
        for v1 in range(election.num_voters):
            for v2 in range(v1 + 1, election.num_voters):
                distances[v1][v2] = swap_distance_between_potes(potes[v1], potes[v2], election.num_candidates)
                distances[v2][v1] = distances[v1][v2]
        election.vote_dists = distances
        return distances


def distances_to_rankings(rankings, distances):
    dists = distances[rankings]
    return np.sum(dists.min(axis=0))


def find_improvement(distances, d, starting, rest, n, k, l):
    for cut in itertools.combinations(range(k), l):
        for paste in itertools.combinations(rest, l):
            ranks = []
            j = 0
            for i in range(k):
                if i in cut:
                    ranks.append(paste[j])
                    j = j + 1
                else:
                    ranks.append(starting[i])
            # check if unique
            if len(set(ranks)) == len(ranks):
                # check if better
                d_new = distances_to_rankings(ranks, distances)
                if d > d_new:
                    return ranks, d_new, True
    return starting, d, False


def local_search_kKemeny_single_k(election, k, l, starting=None) -> dict:

    if starting is None:
        starting = list(range(k))
    distances = get_vote_dists(election)

    n = election.num_voters
    d = distances_to_rankings(starting, distances)
    iter = 0
    check = True
    while (check):
        iter = iter + 1
        rest = [i for i in range(n) if i not in starting]
        for j in range(l):
            starting, d, check = find_improvement(distances, d, starting, rest, n, k, j + 1)
            if check:
                break

    return {'value': d, 'centers': starting}


def _get_distinct_votes_mapping(votes, distinct_votes):
    distinct_votes = [tuple(vote) for vote in distinct_votes]
    mapping = [distinct_votes.index(tuple(vote.tolist())) for vote in votes]
    return mapping


def get_kemeny_score(election, k, method='local_search_domain', search_space=None):

    if method == 'brute_force':
        output = empirical_k_kemeny(election, k, search_space)
    elif method == 'local_search_among_votes':
        output = local_search_kKemeny_single_k(election, k, 1)
    elif method == 'local_search_domain':
        if search_space is None:
            raise ValueError('search_space must be provided for local_search_domain method')
        output = local_search_kKemeny_single_k_from_domain(election, search_space,  k, 1)
    else:
        raise ValueError(
            f"Method {method} is not supported. "
            f"Use 'brute_force', 'local_search_among_votes' or 'local_search_domain'.")

    return output


def get_centers(election_target, k, method, search_space=None):
    output = get_kemeny_score(election_target, k, method, search_space)
    print(output)
    center_ids = []

    for w in output['centers']:
        center_ids.append(int(w))
    center_mapping = [tuple(search_space[center_id]) for center_id in center_ids]

    centers = {'ids': center_ids, 'mapping': center_mapping}
    return centers


def get_clusters(election, centers, search_space):
    cluster_ids = []
    cluster_mapping = {}
    X = []
    # for each point assign the cluster number of the closest k = 4
    distances = vote_domain_dists(election, search_space)
    for i in range(election.num_voters):
        min_dist = float('inf')
        cluster = None
        for j, center in enumerate(centers):
            dist = distances[center][i]
            if dist < min_dist:
                min_dist = dist
                cluster = j+1
        X.append(min_dist)
        if min_dist == 0:
            cluster_ids.append(-cluster)
            cluster_mapping[tuple(election.votes[i])] = -cluster
        else:
            cluster_ids.append(cluster)
            cluster_mapping[tuple(election.votes[i])] = cluster

    clusters = {'ids': cluster_ids, 'mapping': cluster_mapping, 'distances': X}
    return clusters


def empirical_k_kemeny(election, k, search_space=None):
    best_list = None
    min_total_distance = float('inf')

    if search_space is None:
        search_space = list(itertools.permutations(range(election.num_candidates)))

    distances = vote_domain_dists(election, search_space)

    base = [i for i in range(len(search_space))]
    # print(len(search_space))

    for i_vec in tqdm(itertools.combinations(base, k)):
        total_distance = 0
        for j in range(election.num_voters):
            d_vec = tuple([distances[i_vec[i]][j] for i in range(k)])
            value = min(d_vec)
            total_distance += value
        if total_distance < min_total_distance:
            min_total_distance = total_distance
            best_list = tuple(i_vec)
    output ={'value': min_total_distance, 'centers': list(best_list)}
    return output

## k-Kemeny within domain

def vote2pote(vote, m):
    reported = vote[vote != -1]
    part_pote = np.argsort(reported)
    res = []
    i = 0
    non_reported_pos = len(reported) + (m - 1 - len(reported)) / 2
    for c in range(m):
        if c in reported:
            res.append(part_pote[i])
            i = i + 1
        else:
            res.append(non_reported_pos)
    return np.array(res)

def potes_for_domain(domain):
    res = []
    m = len(domain[0])
    for v in domain:
        v = np.array(v)
        res.append(vote2pote(v, m))
    res = np.array(res)
    return res

def vote_domain_dists(election, domain):
    potes_votes = election.get_potes()
    potes_domain = potes_for_domain(domain)
    distances = np.zeros([len(domain), election.num_voters])
    for v in range(election.num_voters):
        for u in range(len(domain)):
            distances[u][v] = swap_distance_between_potes(potes_votes[v],
                                                          potes_domain[u],
                                                          election.num_candidates)
    election.vote_dists = distances
    return distances


def local_search_kKemeny_single_k_from_domain(election, domain, k, l, n_starts=10) -> dict:
    best_result = None
    for _ in range(n_starts):
        starting = random.sample(range(len(domain)), k)
        result = _single_local_search_kKemeny_single_k_from_domain(
            election, domain, k, l, starting=starting)
        if best_result is None or result['value'] < best_result['value']:
            best_result = result
    return best_result


def _single_local_search_kKemeny_single_k_from_domain(election, domain, k, l, starting=None) -> dict:
    if starting is None:
        starting = list(range(k))
    distances = vote_domain_dists(election, domain)

    n = election.num_voters
    dom_size = len(domain)

    if k >= dom_size:
        return {'value': 0, 'centers': [max(dom_size - 1, i) for i in starting]}

    d = distances_to_rankings(starting, distances)
    iter = 0
    check = True
    while (check):
        iter = iter + 1
        rest = [i for i in range(dom_size) if i not in starting]
        for j in range(l):
            starting, d, check = find_improvement(distances, d, starting, rest, n, k, j + 1)
            if check:
                break
    return {'value': d, 'centers': starting}