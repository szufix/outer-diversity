import numpy as np
import itertools
from itertools import product
from itertools import combinations
from collections import Counter
import matplotlib.pyplot as plt

"""
 Generate n random 2D-points (coordinates = floats)
 Params: 
    n:          int             nb of points to generate
    x_range     (float,float)   min and max for x-coordinates
    y_range     (float,float)   min and max for y-coordinates
    seed        int/None        random seed (for reproducibility)

Return: 
    points - list of tuples (x,y) of size n
"""
def generate_random_points(n, x_range=(0, 10), y_range=(0, 10), seed=None):
   
    if seed is not None:
        np.random.seed(seed)

    xs = np.random.uniform(x_range[0], x_range[1], n)
    ys = np.random.uniform(y_range[0], y_range[1], n)
    points = list(zip(xs, ys))
    return points

"""
    Given three points p1, p2, p3, returns the intersection of the L1-bissectors B(p1,p2) \cap B(p1,p3) \cap B(p2,p3). 
    Under L1, this intersection may not exist. This function assumes that it exists -> in theory, it should work even if the intersection 
    does not exist, but never really checked - I check manually the CNS conditions of existence, before calling the function 

    TODO: formal proof that this algorithm works (straightforward but still needed to be written) 
"""
def get_3_intersection(p1, p2, p3, tol=1e-9, eq_tol=1e-6):
    # pi is a (xi,yi) tuple of floats
    points = [p1, p2, p3]
    sols = []

    #Test all ways to break the absolute values (we c)
    for sx in product([1, -1], repeat=3):      # sign choices for x-xj
        for sy in product([1, -1], repeat=3):  # sign choices for y-yj
            # Once we have fixed the signs ( = no more abs values), we have a linear system of 2 equations and 2 unknowns to solve
            A = np.array([[sx[0]-sx[1], sy[0]-sy[1]],
                          [sx[0]-sx[2], sy[0]-sy[2]]], dtype=float)
            b = np.array([
                sx[0]*p1[0] - sx[1]*p2[0] + sy[0]*p1[1] - sy[1]*p2[1],
                sx[0]*p1[0] - sx[2]*p3[0] + sy[0]*p1[1] - sy[2]*p3[1]
            ], dtype=float)

            if abs(np.linalg.det(A)) < tol:
                continue

            x, y = np.linalg.solve(A, b) #(x,y) = a candidate for intersection points

            # check sign consistency
            ok = True
            for j, (xj, yj) in enumerate(points):
                dx = x - xj
                dy = y - yj
                if abs(dx) > tol and (1 if dx > 0 else -1) != sx[j]:
                    ok = False; break
                if abs(dy) > tol and (1 if dy > 0 else -1) != sy[j]:
                    ok = False; break
            if not ok:
                continue

            # final numeric check of equalities 
            d1 = abs(x - p1[0]) + abs(y - p1[1])
            d2 = abs(x - p2[0]) + abs(y - p2[1])
            d3 = abs(x - p3[0]) + abs(y - p3[1])
            if abs(d1 - d2) <= eq_tol and abs(d1 - d3) <= eq_tol:
                sols.append((float(x), float(y)))

    # deduplicate (two cases may give same solution)
    unique = []
    for s in sols:
        if not any(np.hypot(s[0]-t[0], s[1]-t[1]) < 1e-6 for t in unique):
            unique.append(s)

    if len(unique) == 0: #should not happen if we previously check the CNS of existence
        return None
    if len(unique) == 1: #the only case that will happen
        return unique[0]
    return unique   # should not happen if no "special cases" (with random generation, we may assume the points are in general position)


"""
    Given three points p1, p2, p3, checks whether p3 is in the "parallelogram" determined by p1 and p2 
"""
#checks whether p3 is in the parallelogram determined by p1 and p2
def is_in_parallelogram(p1,p2,p3):
    #there are two possible configurations if p1[0] < p2[0]:
    if (p3[1] <= p3[0] - p1[0] + p1[1]) and (p3[1] <= -p3[0] + p2[0] + p2[1]) and (p3[1] >= p3[0] - p2[0] + p2[1]) and (p3[1] >= -p3[0] + p1[0] + p1[1]):
        return True
    if (p3[1] >= p3[0] - p1[0] + p1[1]) and (p3[1] <= -p3[0] + p2[0] + p2[1]) and (p3[1] <= p3[0] - p2[0] + p2[1]) and (p3[1] >= -p3[0] + p1[0] + p1[1]):
        return True
    #... and two more cases if p1[0] > p2[0]:
    if (p3[1] <= p3[0] - p2[0] + p2[1]) and (p3[1] <= -p3[0] + p1[0] + p1[1]) and (p3[1] >= p3[0] - p1[0] + p1[1]) and (p3[1] >= -p3[0] + p2[0] + p2[1]):
        return True
    if (p3[1] >= p3[0] - p2[0] + p2[1]) and (p3[1] <= -p3[0] + p1[0] + p1[1]) and (p3[1] <= p3[0] - p1[0] + p1[1]) and (p3[1] >= -p3[0] + p2[0] + p2[1]):
        return True

    return False 

"""
    Given three points p1, p2, p3, checks whether the bisectors B(p1,p2), B(p1,p3) and B(p2,p3) intersect (in a unique point).
    -> according to Prop. 5 of our paper, they intersect in a unique point iff no one of the points is inside the parallelogram determined by the two others
"""
def exists_3_inter(p1,p2,p3):
    if is_in_parallelogram(p1,p2,p3) or is_in_parallelogram(p2,p3,p1) or is_in_parallelogram(p1,p3,p2):
        return False
    return True     

"""
    Determine all (6) orderings (= rankings) corresponding to surrounding areas of the given 3-intersection point
    Params:
        intersection:     (float,float)     the intersection point of bisectors B(pi,pj), B(pi,pk) and B(pj,pk)
        points:           [(float, float)]  the list of all points
        triple_indices    [int,int,int]     the indices of point pi,pj,pk in the list "points"

    Works as follows:
        - compute distance triple_d of pi (pj,pk) from the intersection (these three points are equidistant)
        - for each point of points (except for pi, pj, pk), computes its distance from the intersection
        - split the points into points closer to intersection than the triple, and further from it than the triple
        - sort each of these parts
        - concatenate them, while putting in between all possible permuations of pi,pj,pk (-> this yields the 6 ranking we are looking for) 

    Return : list of rankings (each ranking = list of ints from 0 to m-1)
"""
def order_points_from_3_inter(intersection, points, triple_indices, tol=1e-9):
    
    xI, yI = intersection
    triple_set = set(triple_indices)

    # Compute the "triple distance" (WLOG from first triple point)
    x0, y0 = points[triple_indices[0]]
    triple_d = abs(xI - x0) + abs(yI - y0)

    closer, further = [], []
    for idx, (x, y) in enumerate(points):
        if idx in triple_set:
            continue
        d = abs(xI - x) + abs(yI - y)
        if d < triple_d - tol:
            closer.append((idx, d))
        else:
            further.append((idx, d))

    # sort closer and further parts
    closer_sorted = [idx for idx, _ in sorted(closer, key=lambda t: t[1])]
    further_sorted = [idx for idx, _ in sorted(further, key=lambda t: t[1])]

    #insert all permutations of the triple in the middle
    orderings = []
    for perm in itertools.permutations(triple_indices):
        ordering = closer_sorted + list(perm) + further_sorted
        orderings.append(ordering)

    return orderings

    """
    For each point, compute the L1 distances to all other points (including itself),
    and return the list of obtained orderings.

    -> necessary, because some areas (= rankings) may be delimited by a bisector that does not intersect with any other bisector
    -> thus, such an area can not be identified as a surrounding area of an intersection
    
    Params: 
    points : list of (x, y)
    tol : float tolerance to break ties deterministically (as we work with floats, normally not necessary)

    TODO: explain formally why this works (easy, but needed to be done explicitely if one day this is made public)
    """
def order_points_when_no_inter(points, tol=1e-9):
  
    n = len(points)
    orderings = []

    for i, (xi, yi) in enumerate(points):
        dists = []
        for j, (xj, yj) in enumerate(points):
            d = abs(xi - xj) + abs(yi - yj)  # L1 distance
            dists.append((j, d))

        # sort by distance, then index to break ties
        sorted_indices = [j for j, _ in sorted(dists, key=lambda t: (t[1], t[0]))]
        orderings.append(sorted_indices)

    return orderings

"""
    The function analogous to order_points_from_3_inter, but this time, we have an intersection
    of B(pi,pj) and B(pk,pl). pi and pj are both of distance d1 from the intersection, and pk and pl of distance d2

    This yields 4 possible surrounding rankings of this intersection.
"""
def order_points_from_2pairs(intersection, points, pair1, pair2, tol=1e-9):
    
    #print(pair1, pair2)
    xI, yI = intersection
    used = set(pair1) | set(pair2)

    # Distances of pairs
    d1 = abs(xI - points[pair1[0]][0]) + abs(yI - points[pair1[0]][1])
    d2 = abs(xI - points[pair2[0]][0]) + abs(yI - points[pair2[0]][1])

    closer, middle, further = [], [], []
    for idx, (x, y) in enumerate(points):
        if idx in used:
            continue
        d = abs(xI - x) + abs(yI - y)
        if d < min(d1, d2) - tol:
            closer.append((idx, d))
        elif d > max(d1, d2) + tol:
            further.append((idx, d))
        else:
            middle.append((idx, d))

    closer_sorted = [i for i, _ in sorted(closer, key=lambda t: t[1])]
    #print(closer_sorted)
    middle_sorted = [i for i, _ in sorted(middle, key=lambda t: t[1])]
    #print(middle_sorted)
    further_sorted = [i for i, _ in sorted(further, key=lambda t: t[1])]
    #print(further_sorted)

    orderings = []

    if d1 < d2:
        # pair1 closer than pair2
        for perm1 in itertools.permutations(pair1, 2):
            for perm2 in itertools.permutations(pair2, 2):
                ordering = closer_sorted + list(perm1) + middle_sorted + list(perm2) + further_sorted
                #print(ordering)
                orderings.append(ordering)
    else:
        # pair2 closer than pair1
        for perm2 in itertools.permutations(pair2, 2):
            for perm1 in itertools.permutations(pair1, 2):
                ordering = closer_sorted + list(perm2) + middle_sorted + list(perm1) + further_sorted
                #print(ordering)
                orderings.append(ordering)

    return orderings

"""
    Given three points p1, p2, p3, returns the intersection of the L1-bissectors B(p1,p2) \cap B(p1,p3) \cap B(p2,p3). 
    Under L1, this intersection may not exist. This function assumes that it exists -> in theory, it should work even if the intersection 
    does not exist, but never really checked - I check manually the CNS conditions of existence, before calling the function 

    TODO: formal proof that this algorithm works (straightforward but still needed to be written) 
"""

""" 
    Analogous to the get_3_intersection function, this time to compute intersection(s) (1 or 2 possible) 
    between B(p1,p2) and B(p3,p4). 

    Still the same BF-like principle - we try to break inequalities in any possible way, and then check consistency. 

    Originally, I assumed that the intersection exists (but it also works when it does not exist, and I finally use it as so in the simulations...)

    TODO: proof formally it works (again easy, but needed to be done properly)

"""
def get_4_inter(p_i, p_j, p_k, p_l, tol=1e-9, eq_tol=1e-6):
    
    pts = [p_i, p_j, p_k, p_l]
    sols = []

    # sx[p] is sign of (x - x_p) in {+1,-1}, sy[p] sign of (y - y_p)
    # enumerate all sign possibilities  -> a bit BF, but works fairly fast, so I keep this strategy for the moment
    for sx in product([1, -1], repeat=4):
        for sy in product([1, -1], repeat=4):
            # build linear system once the abs are broken
            A = np.array([
                [sx[0] - sx[1], sy[0] - sy[1]],
                [sx[2] - sx[3], sy[2] - sy[3]]
            ], dtype=float)

            b = np.array([
                sx[0]*p_i[0] - sx[1]*p_j[0] + sy[0]*p_i[1] - sy[1]*p_j[1],
                sx[2]*p_k[0] - sx[3]*p_l[0] + sy[2]*p_k[1] - sy[3]*p_l[1]
            ], dtype=float)

            detA = A[0,0]*A[1,1] - A[0,1]*A[1,0]
            if abs(detA) < tol:
                # degenerate: either no unique solution for this sign-case (parallel lines)
                # skip this sign-case
                continue

            x_val, y_val = np.linalg.solve(A, b)

            # check consistency of assumed signs for all four points
            ok = True
            for idx, (xp, yp) in enumerate(pts):
                dx = x_val - xp
                dy = y_val - yp

                if abs(dx) > tol:
                    sdx = 1 if dx > 0 else -1
                    if sdx != sx[idx]:
                        ok = False
                        break
                if abs(dy) > tol:
                    sdy = 1 if dy > 0 else -1
                    if sdy != sy[idx]:
                        ok = False
                        break
            if not ok:
                continue

            # final numeric check of equalities
            d_ij_left = abs(x_val - p_i[0]) + abs(y_val - p_i[1])
            d_ij_right = abs(x_val - p_j[0]) + abs(y_val - p_j[1])
            d_kl_left = abs(x_val - p_k[0]) + abs(y_val - p_k[1])
            d_kl_right = abs(x_val - p_l[0]) + abs(y_val - p_l[1])

            if (abs(d_ij_left - d_ij_right) <= eq_tol and
                abs(d_kl_left - d_kl_right) <= eq_tol):
                sols.append((float(x_val), float(y_val)))

    # deduplicate (two sign patterns can yield the same point)
    unique = []
    for s in sols:
        if not any(np.hypot(s[0]-t[0], s[1]-t[1]) < 1e-7 for t in unique):
            unique.append(s)

    return unique

"""
    Given 4 points p1,p2,p3,p4, check whether at least one intersection of B(p1,p2) and B(p3,p4) exists
"""
def exists_4_inter(p1,p2,p3,p4):
    # 2 horizontals:
    if (abs(p1[0] - p2[0]) < abs(p1[1] - p2[1])) and (abs(p3[0] - p4[0]) < abs(p3[1] - p4[1])):
        d12 = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        upper12 = min(p1[1],p2[1]) + d12/2.0
        lower12 = max(p1[1],p2[1]) - d12/2.0 

        d34 = abs(p3[0] - p4[0]) + abs(p3[1] - p4[1])
        upper34 = min(p3[1],p4[1]) + d34/2.0
        lower34 = max(p3[1],p4[1]) - d34/2.0 

        if upper12 < lower34 or upper34 < lower12:
            return False
    #2 verticals:
    if (abs(p1[0] - p2[0]) > abs(p1[1] - p2[1])) and (abs(p3[0] - p4[0]) > abs(p3[1] - p4[1])):
        d12 = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        right12 = min(p1[0],p2[0]) + d12/2.0
        left12 = max(p1[0],p2[0]) - d12/2.0 

        d34 = abs(p3[0] - p4[0]) + abs(p3[1] - p4[1])
        right34 = min(p3[0],p4[0]) + d34/2.0
        left34 = max(p3[0],p4[0]) - d34/2.0 

        if right12 < left34 or right34 < left12:
            return False
    
    #otherwise - 1H + 1V, so necessarily intersect:
    return True

"""
#############################################

Many auxiliary functions to list all maximum profiles (in sense of inclusion)
-> we need to know to check, given two profiles, whether the smaller one is embedded in the bigger one *up to renaming the points*
"""

"""Return permutation (as dict) mapping from ordA to ordB."""
def mapping_from_two_orderings(ordA, ordB):
    return {a: b for a, b in zip(ordA, ordB)}

"""Apply mapping to one ordering."""
def apply_mapping(ordering, mapping):
    return tuple(mapping[x] for x in ordering)

"""Apply mapping to a whole profile."""
def relabel_profile(profile, mapping):
    return [apply_mapping(ord_, mapping) for ord_ in profile]

""" Check if two profiles are the same up to renaming the points."""
def profiles_equivalent(profileA, profileB):
    if len(profileA) != len(profileB):
        return False
    setB = set(tuple(ord_) for ord_ in profileB)
    ordA0 = profileA[0]
    for ordB in profileB:
        mapping = mapping_from_two_orderings(ordA0, ordB)
        relabeledA = relabel_profile(profileA, mapping)
        if set(relabeledA) == setB:
            return True
    return False

"""
    Check if profileA embeds into profileB up to renaming the points.
"""
def is_embedded(profileA, profileB):
    if len(profileA) > len(profileB):
        return False
    setB = set(tuple(ord_) for ord_ in profileB)
    ordA0 = profileA[0]
    for ordB in profileB:
        mapping = mapping_from_two_orderings(ordA0, ordB)
        relabeledA = relabel_profile(profileA, mapping)
        if all(tuple(ord_) in setB for ord_ in relabeledA):
            return True
    return False

"""
    Insert new_profile into dict_profiles with pruning:
    - Skip if embedded in existing profile
    - Otherwise add it
    - Remove existing profiles that are embedded in the new one
"""
def add_profile_with_pruning(dict_profiles, new_profile):
   
    k = len(new_profile)
    #print("new: ", new_profile)
    #print("current dict: ", dict_profiles)

    # 1. Skip if embedded
    for k2, profiles in list(dict_profiles.items()):
        for prof in profiles:
            if is_embedded(new_profile, prof):
                #print("embedded in ", prof)
                return  # already contained, skip

    # 2. Remove profiles that are embedded in new_profile
    for k2, profiles in list(dict_profiles.items()):
        to_keep = []
        for prof in profiles:
            if not is_embedded(prof, new_profile):
                to_keep.append(prof)
        dict_profiles[k2] = to_keep

    # Finally add the new profile
    dict_profiles.setdefault(len(new_profile), []).append(new_profile)

"""
###############################################
The end of checking whether one profile is a subprofile of another one, up to renaming the candidates
"""

"""
    I was wondering what is a typical size of a random L1-profile (if I generate the points uniformly at random)
    -> here, I do not check for maximum profile, but simply for all profiles
    (eg., if I have generated 1000 instances, I want to see the distribution of their sizes, I do not care whether these are maximum or not) 

    counter : for each k, the number of randomly generated profiles of size k
    m: the number of candidates
"""
def plot_histogram_random_size(counter, m):
    data = []
    for k, count in counter.items():
        data.extend([k] * count)
    plt.hist(data, bins=range(min(data), max(data)+2), align="left", rwidth=0.8)
    plt.xlabel(f"Number of different rankings")
    plt.ylabel("Nb of profiles")
    plt.title(f"Size of randomly generated profile (m = {m})")
    plt.show()

"""
    Plot histogram of maximum profiles by size,
    print the counts in terminal.

    Here, dict_profiles contains only maximum profiles (in sense of inclusion), I want to know how many there are macimum profiles of each size...
"""
def plot_max_profiles_histogram(dict_profiles,m):
    
    ks = sorted(dict_profiles.keys())
    counts = [len(dict_profiles[k]) for k in ks]
    total = sum(counts)

    plt.bar(ks, counts, width=0.8, align="center")
    plt.xlabel("Size of profile (# different rankings)")
    plt.ylabel("Number of maximum profiles")
    plt.title(f"Maximal profiles distribution (total = {total}, m = {m})")
    plt.show()

    print(f"Total maximum profiles: {total}")
    print("\nSize : Count ")
    for k, count in zip(ks, counts):
        print(f"{k:6} : {count:5}")



#Some tests with concrete instances of our paper - just for a check

#points = [(4,1), (1,6), (6,8), (8,2)] 
#points = [(0,8), (10,10),(4,1),(8,3)]

m = 5 #number of points/candidates 

size_random = Counter() #size_random[k] counts the number of randomly generated profiles (maximum or not in sense of inclusion) of size k
nb_tests = 1000 #the number of instances to generate
dict_profiles = {} #the dictionnary of maximum profiles -> for each k, dict_profiles[k] constains the list of maximum profiles of size k ; each profile is a list of rankings (ranking = a list of indices from 0 to m-1) 

for i in range(nb_tests):
    points = generate_random_points(m,x_range = (0,20), y_range=(0,20))
    #print(points)

    profile = []

    # detecting all rankings corresponding to areas surrounding some 3-intersection (B(pi,pj), B(pi,pk) and B(pj,pk)) 
    for p1,p2,p3 in combinations(points,3):
        if exists_3_inter(p1,p2,p3):
            inter = get_3_intersection(p1, p2, p3)
            #print("Triple: ",p1, p2,p3, " inter: ", inter)
            triple_indices = (points.index(p1), points.index(p2), points.index(p3))
            profile += order_points_from_3_inter(inter, points, triple_indices)
            #print(profile)
        #else:
            #print("Triple: ", p1,p2,p3, " does not intersect")

    # detecting all rankings corresponding to areas surrounding some intersection of B(pi,pj) and B(pk,pl)
    for a, b, c, d in combinations(range(len(points)), 4):
        # 4 distinct points, 3 ways to split them into pairs
        pairs = [((a, b), (c, d)),((a, c), (b, d)),((a, d), (b, c))]
    
        for (i, j), (k, l) in pairs:
            inters = get_4_inter(points[i], points[j],points[k], points[l])
            for inter in inters:
                #print("pairs: ", i,j, " and ", k,l, " inter: ", inter)
                profile += order_points_from_2pairs(inter,points,(i,j),(k,l))

    #Case when there are no intersections, or some "isolated areas" not captured on the intersections: 
    profile += order_points_when_no_inter(points)

    #Removing duplicates (some rankings = areas are adjacent to several intersections, so detected twice)
    profile = list(map(list, dict.fromkeys(map(tuple, profile))))

    #check whether the profile is maximum, and if so add it to the disctionary of maximum profile 
    add_profile_with_pruning(dict_profiles, profile)

    #print(profile)
    #print("Profil size: ",len(profile))
    size_random[len(profile)] +=1 

#print(dict_profiles)

#print(size_random)

plot_histogram_random_size(size_random,m)
plot_max_profiles_histogram(dict_profiles,m)