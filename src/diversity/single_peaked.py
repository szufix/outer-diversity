def distance_vote_single_peaked_domain(vote):
    """
    Compute the distance between a vote and a single-peaked domain.

    WE ASSUME THE 0,1,2,3 AXIS !!!

    Args:
        vote: List representing the vote ordering (indices of candidates in preference order)

    Returns:
        int: Minimum distance between the vote and single-peaked domain
    """


    m = len(vote)
    pote = [list(vote).index(i) for i, _ in enumerate(vote)]

    # PHASE 1: PRECOMPUTATION
    # Initialize L and R matrices
    L = [[0] * (m+1) for _ in range(m+1)]
    R = [[0] * (m+1) for _ in range(m+1)]

    for i in range(1, m+1):
        L[i][i] = 0
        R[i][i] = 0

        # Compute L[i][j] for j > i
        for j in range(i + 1, m + 1):

            crossing = int(pote[i-1] < pote[j-1])

            L[i][j] = L[i][j - 1] + crossing

        # Compute R[j][i] for j < i
        for j in reversed(range(1, i)):

            crossing = int(pote[i-1] < pote[j-1])

            R[j][i] = R[j + 1][i] + crossing

    # PHASE 2: MAIN COMPUTATION
    # Initialize DP table
    D = [[float('inf')] * (m + 1) for _ in range(m)]

    # Base case
    D[0][0] = 0

    # Fill first column
    for l in range(1, m):
        D[l][0] = D[l - 1][0] + L[l][m]

    # Fill the DP table
    for r in range(1, m):
        # First row for each r
        D[0][r] = D[0][r - 1] + R[1][m + 1 - r]

        # Fill remaining entries
        for l in range(1, m - r):
            option1 = D[l - 1][r] + L[l][m - r] if m - r >= 0 else float('inf')
            option2 = D[l][r - 1] + R[l + 1][m + 1 - r] if m + 1 - r >= 0 else float('inf')
            D[l][r] = min(option1, option2)

    # Return the minimum over valid final states
    result = float('inf')
    for l in range(1, m+1):
        if m - l >= 0:
            result = min(result, D[l - 1][m - l])

    return result

