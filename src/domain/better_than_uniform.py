import numpy as np

from src.domain import single_crossing_domain


#################################################


def single_crossing_with_gaps(num_voters, num_candidates, with_domain=False):
    domain = single_crossing_domain(num_candidates)
    # domain = [tuple(x) for x in domain]
    mask = [0 for _ in range(len(domain))]
    gap = 12
    ctr = 1
    round = 1

    while ctr < len(domain):
        for i in range(round):
            mask[ctr] = 1
            ctr += 1
            if ctr >= len(domain):
                break
        round *= 2
        ctr += gap
    mask = np.array(mask)
    mask = mask / np.sum(mask)
    indices = np.random.choice(len(domain), size=num_voters, p=mask)
    sampled_votes = [domain[i] for i in indices]
    if with_domain:
        return sampled_votes, domain
    else:
        return sampled_votes


def weighted_single_crossing_with_gaps(num_voters, num_candidates, with_domain=False):
    domain = single_crossing_domain(num_candidates)
    # domain = [tuple(x) for x in domain]
    mask = [0 for _ in range(len(domain))]
    gap = 12
    ctr = 0
    round = 1
    power = 1

    while ctr < len(domain):
        for i in range(round):
            mask[ctr] = power
            ctr += 1
            if ctr >= len(domain):
                break
        power /= 2
        round *= 2
        ctr += gap

    mask = np.array(mask)
    mask = mask / np.sum(mask)
    indices = np.random.choice(len(domain), size=num_voters, p=mask)
    sampled_votes = [domain[i] for i in indices]


    if with_domain:
        return sampled_votes, domain
    else:
        return sampled_votes


# weighted_single_crossing_with_gaps(10, 16)

