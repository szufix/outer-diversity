data = {}

data['2'] = """ 
12 21 
"""

data['3'] = """ 
123 132 312 321 
"""

data['4'] = """ 
1234 1324 3124 1342 3142 3412 4312 3421 4321 
"""

data['5'] = """
12345 12435 12453 21345 21435 21453 24135 24153 24513 24531 
42135 42153 42513 42531 45213 45231 45321 54213 54231 54321
"""

data['6'] = """
123456 123465 124356 124365 124635 124653 213456 213465 214356 214365
214635 214653 241356 241365 241635 241653 246135 246153 246513 246531 
421356 421365 421635 421653 426135 426153 426513 426531 462135 462153
462513 462531 465213 465231 465321 642135 642153 642513 642531 645213 
645231 645321 654213 654231 654321
"""

data['7'] = """
1234567 1234657 1234675 1243567 1243657 1243675 1246357 1246375 1246735 1246753 
2134567 2134657 2134675 2143567 2143657 2143675 2146357 2146375 2146735 2146753
2413567 2413657 2413675 2416357 2416375 2416735 2416753 2461357 2461375 2461735
2461753 2467135 2467153 2467513 2467531 4213567 4213657 4213675 4216357 4216375 
4216735 4216753 4261357 4261375 4261735 4261753 4267135 4267153 4267513 4267531
4621357 4621375 4621735 4621753 4627135 4627153 4627513 4627531 4672135 4672153
4672513 4672531 4675213 4675231 4675321 6421357 6421375 6421735 6421753 6427135
6427153 6427513 6427531 6472135 6472153 6472513 6472531 6475213 6475231 6475321
6742135 6742153 6742513 6742531 6745213 6745231 6745321 6754213 6754231 6754321
7642135 7642153 7642513 7642531 7645213 7645231 7645321 7654213 7654231 7654321
"""

data['8'] = """
12345678 12345687 12345867 12345876 12346578 12346587 12346758 12346785 12354678 12354687
12354867 12354876 12358467 12358476 12364578 12364587 12364758 12364785 12367458 12367485
12435678 12435687 12435867 12435876 12436578 12436587 12436758 12436785 12453678 12453687
12453867 12453876 12458367 12458376 12463578 12463587 12463758 12463785 12467358 12467385
14235678 14235687 14235867 14235876 14236578 14236587 14236758 14236785 14253678 14253687
14253867 14253876 14258367 14258376 14263578 14263587 14263758 14263785 14267358 14267385
14523678 14523687 14523867 14523876 14528367 14528376 14582367 14582376 14623578 14623587
14623758 14623785 14627358 14627385 14672358 14672385 21345678 21345687 21345867 21345876
21346578 21346587 21346758 21346785 21354678 21354687 21354867 21354876 21358467 21358476
21364578 21364587 21364758 21364785 21367458 21367485 21435678 21435687 21435867 21435876
21436578 21436587 21436758 21436785 21453678 21453687 21453867 21453876 21458367 21458376
21463578 21463587 21463758 21463785 21467358 21467385 23145678 23145687 23145867 23145876
23146578 23146587 23146758 23146785 23154678 23154687 23154867 23154876 23158467 23158476
23164578 23164587 23164758 23164785 23167458 23167485 23514678 23514687 23514867 23514876
23518467 23518476 23581467 23581476 23614578 23614587 23614758 23614785 23617458 23617485
23671458 23671485 32145678 32145687 32145867 32145876 32146578 32146587 32146758 32146785
32154678 32154687 32154867 32154876 32158467 32158476 32164578 32164587 32164758 32164785
32167458 32167485 32514678 32514687 32514867 32514876 32518467 32518476 32581467 32581476
32614578 32614587 32614758 32614785 32617458 32617485 32671458 32671485 41235678 41235687
41235867 41235876 41236578 41236587 41236758 41236785 41253678 41253687 41253867 41253876
41258367 41258376 41263578 41263587 41263758 41263785 41267358 41267385 41523678 41523687
41523867 41523876 41528367 41528376 41582367 41582376 41623578 41623587 41623758 41623785
41627358 41627385 41672358 41672385
"""


from itertools import combinations
from typing import List, Sequence, Hashable, Tuple, Optional, Dict

Order = Sequence[Hashable]          # one linear order, e.g. [1,2,3,4]
Domain = Sequence[Order]            # a domain is a list of such orders


from itertools import permutations, combinations

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
    alts = list(range(1, m + 1))
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

    return domain


def as_strings(domain):
    """Convert tuples like (1,2,3) to '123'."""
    return ["".join(str(a) for a in order) for order in domain]



def is_condorcet_domain(domain: Domain, return_witness: bool = True):
    """
    Check whether a given domain of linear orders is a Condorcet domain.

    Parameters
    ----------
    domain : list of orders
        Each order is a sequence (list/tuple) of distinct, hashable alternatives.
        All orders should be permutations of the same alternative set.
    return_witness : bool
        If True and the domain is NOT Condorcet, return a witness:
        a triple of alternatives and three orders that generate a cycle.

    Returns
    -------
    is_cd : bool
        True iff the domain is a Condorcet domain.
    witness : dict or None
        If is_cd is False and return_witness is True, a dictionary:
        {
          'triple': (a, b, c),
          'pattern_codes': ['123', '231', '312'],  # or the reverse set
          'orders': [order1, order2, order3]
        }
        Otherwise, None.
    """
    if not domain:
        raise ValueError("Domain must contain at least one order.")

    # --- 1. collect and sanity-check alternatives ---
    alts = set(domain[0])
    for order in domain:
        if set(order) != alts or len(order) != len(alts):
            raise ValueError("All orders must be permutations of the same set of alternatives.")

    m = len(alts)
    if m < 3:
        # With fewer than 3 alternatives, majority is always transitive.
        return True, None if return_witness else True

    # Forbidden triplets of rankings on {1,2,3} that generate a Condorcet cycle:
    cycle_patterns = [
        {"123", "231", "312"},  # a > b > c, b > c > a, c > a > b
        {"132", "321", "213"},  # a > c > b, c > b > a, b > a > c
    ]

    # --- 2. scan all triples of alternatives ---
    for (a, b, c) in combinations(alts, 3):
        # Map these three alternatives onto 1,2,3 to normalize patterns
        alt_to_symbol: Dict[Hashable, str] = {a: "1", b: "2", c: "3"}

        # Collect the normalized permutations (strings like "123", "213", etc.)
        patterns_for_triple = set()
        order_code_map: Dict[str, Order] = {}

        for order in domain:
            # Restrict the order to this triple
            restricted = [x for x in order if x in (a, b, c)]
            # Convert to a '123'-style code
            code = "".join(alt_to_symbol[x] for x in restricted)
            patterns_for_triple.add(code)
            # Save some order that realizes this code (for witness)
            if code not in order_code_map:
                order_code_map[code] = order

        # --- 3. check whether we contain a cycle-generating pattern set ---
        for forbidden in cycle_patterns:
            if forbidden.issubset(patterns_for_triple):
                if not return_witness:
                    return False, None

                # Construct a witness
                witness_orders = [order_code_map[code] for code in forbidden]
                witness = {
                    "triple": (a, b, c),
                    "pattern_codes": sorted(list(forbidden)),
                    "orders": witness_orders,
                }
                return False, witness

    # If we never found a bad triple, it's a Condorcet domain
    return True, None if return_witness else True



def largest_condorcet_domain(num_candidates):
    """
    Generate the largest Condorcet domain for a given number of candidates.

    Parameters
    ----------
        num_candidates : int
            Number of candidates.

    Returns
    -------
        List[List[int]]
            List of votes in the largest Condorcet domain.
    """
    if num_candidates not in [2,3,4,5,6,7,8]:
        raise NotImplementedError("Largest Condorcet domain is only implemented for "
                                  "2,3,4,5,6,7, and 8 candidates.")

    # Split into numbers
    numbers = data[str(num_candidates)].split()

    # Convert each number into a list of digits
    votes = [[int(d)-1 for d in num] for num in numbers]

    return votes


# for num_candidates in [2,3,4,5,6,7,8]:
#     domain = largest_condorcet_domain(num_candidates)
#     is_cd, witness = is_condorcet_domain(domain, return_witness=True)
#     assert is_cd, f"Domain for {num_candidates} candidates is not Condorcet!"
#     print(f"Largest Condorcet domain for {num_candidates} candidates has {len(domain)} votes.")