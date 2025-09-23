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


def distance_vote_single_peaked_on_circle_domain(vote):
    """
    Compute the distance between a vote and a single-peaked-on-a-circle domain.

    WE ASSUME THE 0,1,2,3,...,m-1 CIRCULAR AXIS !!!

    Args:
        vote: List representing the vote ordering (indices of candidates in preference order)

    Returns:
        int: Minimum distance between the vote and single-peaked-on-a-circle domain
    """
    
    m = len(vote)
    pote = [list(vote).index(i) for i, _ in enumerate(vote)]
    
    # Phase 1: Precomputation
    # The algorithm extends the axis: c_{i+m} = c_i for i ∈ [m]
    # We interpret indices modulo m for the circular property
    
    # Initialize L and R matrices
    # We'll use a larger size to accommodate the algorithm's indexing scheme
    L = [[0] * (2 * m) for _ in range(2 * m)]
    R = [[0] * (2 * m) for _ in range(2 * m)]
    
    # Algorithm uses 1-indexed, but we implement 0-indexed
    # i ∈ [m] in algorithm means i from 1 to m, which we do as i from 0 to m-1
    for i in range(m):
        # Initialize diagonal elements: L_{i,i} = 0, R_{i+m,i} = 0
        L[i%m][i%m] = 0
        R[(i + m)%m][i%m] = 0
        
        # Compute L_{i,i+j} for j ∈ [m-1] (j from 1 to m-1)
        for j in range(1, m):
            # c_{i+j} corresponds to candidate (i+j) % m due to circular property
            candidate_i = i
            candidate_i_plus_j = (i + j) % m

            # Check if c_{i+j} ≻_v c_i
            crossing = int(pote[candidate_i_plus_j] < pote[candidate_i])
            L[i%m][(i + j)%m] = L[i%m][(i + j - 1)%m] + crossing
        
        # Compute R_{i+m-j,i} for j ∈ [m-1] (j from 1 to m-1)
        # This iterates j from 1 to m-1, so we compute R_{i+m-1,i}, R_{i+m-2,i}, ..., R_{i+1,i}
        for j in range(1, m):
            idx = i + m - j
            if 0 <= idx < 2 * m and idx + 1 < 2 * m:
                # c_{i+m-j} corresponds to candidate (i+m-j) % m = (i-j) % m
                candidate_i = i
                candidate_i_minus_j = (i - j) % m
                
                # Check if c_{i+m-j} ≻_v c_i
                crossing = int(pote[candidate_i_minus_j] < pote[candidate_i])
                R[idx%m][i%m] = R[(idx + 1)%m][i%m] + crossing
    
    # Phase 2: Main Computation
    # Initialize DP table
    D = [[0] * (2 * m) for _ in range(2 * m)]
    
    # Initialize base cases: D_{i,i} = L_{i,i+m-1} for i ∈ [m]
    for i in range(m):
        if i + m - 1 < 2 * m:
            D[i%m][i%m] = L[i%m][(i + m - 1)%m]
    
    # Fill the DP table for r ∈ [m-2] (r from 1 to m-2)
    for r in range(1, m - 1):
        for i in range(m):
            if i + r < 2 * m:
                # D_{i,i+r} = min(D_{i,i+r-1} + L_{i+r,i+m-1}, D_{i+1,i+r} + R_{i+r+1,i})
                option1 = float('inf')
                option2 = float('inf')
                
                # First option: D_{i,i+r-1} + L_{i+r,i+m-1}
                option1 = D[i%m][(i + r - 1)%m] + L[(i + r)%m][(i + m - 1)%m]
                
                # Second option: D_{i+1,i+r} + R_{i+r+1,i}
                # Handle wrap-around for i+1
                next_i = (i + 1) % m

                option2 = D[next_i%m][(i + r)%m] + R[(i + r + 1)%m][i%m]
                D[i%m][(i + r)%m] = min(option1, option2)
    
    # Return min_{i ∈ [m]} D_{i,i+m-2}
    result = float('inf')
    for i in range(m):
        if i + m - 2 < 2 * m:
            result = min(result, D[i%m][(i + m - 2)%m])
    
    return result