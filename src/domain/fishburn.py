from itertools import permutations, combinations


def is_fishburn_domain(domain):
    """
    Check whether a given domain (list of votes) satisfies Fishburn's
    alternating triple rule used to construct the Fishburn domain.

    The domain is a list (or iterable) of votes, where each vote is an
    ordering of alternatives represented by integers/labels. We assume all
    votes use the same set of alternatives. The rule checked is:

      For every triple of distinct alternatives a < b < c (using the numeric
      ordering of labels), if b is odd then b is never top among {a,b,c}
      in any vote; if b is even then b is never bottom among {a,b,c} in any
      vote.

    Returns True if the domain satisfies the rule, False otherwise.
    """
    # domain must be non-empty and votes must be sequences
    domain = list(domain)
    if len(domain) == 0:
        return True  # empty domain trivially satisfies the condition

    # infer alternatives from first vote
    first = domain[0]
    try:
        alts = list(first)
    except TypeError:
        raise ValueError("Votes in domain must be sequences of alternatives")

    # Map alternative labels to canonical numeric ordering for triple test.
    # The fishburn construction in this module uses 0..m-1 ordering. We will
    # sort the alternative labels by their natural ordering and use that
    # ordering to determine triples a<b<c.
    sorted_alts = sorted(alts)

    # create index mapping for each vote and validate votes contain same alts
    m = len(sorted_alts)
    expected_set = set(sorted_alts)

    for v in domain:
        if set(v) != expected_set:
            raise ValueError("All votes in domain must contain the same set of alternatives")

    # check each vote against the triple rule
    for vote in domain:
        pos = {a: i for i, a in enumerate(vote)}
        # iterate over triples using sorted_alts
        for i_idx in range(m - 2):
            for j_idx in range(i_idx + 1, m - 1):
                for k_idx in range(j_idx + 1, m):
                    i = sorted_alts[i_idx]
                    j = sorted_alts[j_idx]
                    k = sorted_alts[k_idx]

                    pi, pj, pk = pos[i], pos[j], pos[k]

                    # For parity, interpret alternatives as integers if possible.
                    # If labels are not integers, fall back to their position parity
                    # in the sorted order (to mimic the module's 0..m-1 parity).
                    try:
                        j_label = int(j)
                        parity = j_label % 2
                    except Exception:
                        parity = j_idx % 2

                    if parity == 1:
                        # j is odd: NEVER-TOP
                        if pj < pi and pj < pk:
                            return False
                    else:
                        # j is even: NEVER-BOTTOM
                        if pj > pi and pj > pk:
                            return False

    return True


def fishburn_alternating_domain(m):
    """
    Construct Fishburn's alternating-scheme Condorcet domain F_m
    on alternatives {1, ..., m}, using the rule:

      - for every triple i < j < k:
          * if j is odd:  j is NEVER TOP in {i, j, k}
          * if j is even: j is NEVER BOTTOM in {i, j, k}

    Returns:
        A list of permutations (each a tuple) of {1, ..., m}.
    """
    alts = list(range(0, m))
    domain = []

    for p in permutations(alts):
        pos = {a: i for i, a in enumerate(p)}
        ok = True

        # check all triples i<j<k
        for i, j, k in combinations(alts, 3):
            if not (i < j < k):
                continue

            pi, pj, pk = pos[i], pos[j], pos[k]

            if j % 2 == 1:
                # j is odd: NEVER-TOP condition on triple {i,j,k}
                # so j cannot be top among {i,j,k}
                if pj < pi and pj < pk:
                    ok = False
                    break
            else:
                # j is even: NEVER-BOTTOM condition on triple {i,j,k}
                # so j cannot be bottom among {i,j,k}
                if pj > pi and pj > pk:
                    ok = False
                    break

        if ok:
            domain.append(p)

    # convert domain to list of lists
    domain = [list(vote) for vote in domain]

    return domain


def largest_fishburn_domain(num_candidates):
    """
    Generate the largest Fishburn domain for a given number of candidates.

    Parameters
    ----------
        num_candidates : int
            Number of candidates.

    Returns
    -------
        List[List[int]]
            List of votes in the largest Condorcet domain.
    """

    return fishburn_alternating_domain(num_candidates)

