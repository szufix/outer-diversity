from gurobipy import Model, GRB, LinExpr, QuadExpr
from random import *
from src.domain.validator import validate_domain
from src.domain.extenders import extend_to_reversed

import warnings
from src.domain.sizes import (
    EUC_1D_DOMAIN_SIZE,
    EUC_2D_DOMAIN_SIZE,
)

def c_pref_d_gurobi(model, v, c, d):
    """Creates inequality saying that voter v prefers c to d."""
    expr_c = QuadExpr()
    expr_d = QuadExpr()
    for i in range(len(c)):
        expr_c.add(-2 * c[i] * v[i])
        expr_c.add(c[i] ** 2)

        expr_d.add(-2 * d[i] * v[i])
        expr_d.add(d[i] ** 2)

    model.addConstr(expr_c <= expr_d)

def possible_vote_gurobi(C, vote, bounding_box=False):
    """Check if vote can be implemented given candidate locations."""
    d = len(C[0])
    model = Model("votecheck")
    model.setParam("OutputFlag", 0)  # Suppress Gurobi output

    v = [model.addVar(lb=-GRB.INFINITY, name=f"x{i}") for i in range(d)]
    model.update()

    m = len(vote)
    for i in range(m):
        for j in range(i + 1, m):
            c_pref_d_gurobi(model, v, C[vote[i]], C[vote[j]])

    if bounding_box is not False:
        for i in range(d):
            model.addConstr(v[i] >= bounding_box[0])
            model.addConstr(v[i] <= bounding_box[1])

    model.setObjective(0, GRB.MINIMIZE)
    model.optimize()

    return model.status == GRB.OPTIMAL

def euclidean_domain_gurobi(C, bounding_box=False):
    """Outputs all possible votes in the Euclidean domain."""
    m = len(C)
    V = [[0]]
    for cand in range(1, m):
        newV = []
        for v in V:
            for i in range(cand + 1):
                newv = v[:i] + [cand] + v[i:]
                if possible_vote_gurobi(C, newv, bounding_box):
                    newV.append(newv)
        V = newV

    return V

@validate_domain
def euclidean_1d_domain(num_candidates, bounds=False):
    bounding_box = (bounds[0], bounds[1]) if bounds else False

    domain = euclidean_domain_gurobi(
        [[random()] for _ in range(num_candidates)], bounding_box=bounding_box)
    if len(domain) < EUC_1D_DOMAIN_SIZE[num_candidates]:
            warnings.warn(f"Warning: Domain size {len(domain)} "
                          f"smaller than expected size {EUC_1D_DOMAIN_SIZE[num_candidates]}.")
    return domain

@validate_domain
def euclidean_2d_domain(num_candidates, bounds=False):
    bounding_box = (bounds[0], bounds[2],
                    bounds[1], bounds[3]) if bounds else False

    domain = euclidean_domain_gurobi(
        [[random(), random()] for _ in range(num_candidates)], bounding_box=bounding_box)
    if not bounds and len(domain) < EUC_2D_DOMAIN_SIZE[num_candidates]:
            warnings.warn(f"Warning: Domain size {len(domain)} "
                          f"smaller than expected size {EUC_2D_DOMAIN_SIZE[num_candidates]}.")
    return domain

@validate_domain
def euclidean_3d_domain(num_candidates, bounds=False):

    bounding_box = (bounds[0], bounds[3],
                    bounds[1], bounds[4],
                    bounds[2], bounds[5]) if bounds else False

    return euclidean_domain_gurobi(
        [[random(), random(), random()]
         for _ in range(num_candidates)], bounding_box=bounding_box)

@validate_domain
def euclidean_4d_domain(num_candidates, bounds=False):

    bounding_box = (bounds[0], bounds[4],
                    bounds[1], bounds[5],
                    bounds[2], bounds[6],
                    bounds[3], bounds[7]) if bounds else False

    return euclidean_domain_gurobi(
        [[random(), random(), random(), random()]
         for _ in range(num_candidates)], bounding_box=bounding_box)


@validate_domain
def euclidean_5d_domain(num_candidates, bounds=False):

    bounding_box = (bounds[0], bounds[5],
                    bounds[1], bounds[6],
                    bounds[2], bounds[7],
                    bounds[3], bounds[8],
                    bounds[4], bounds[9]) if bounds else False

    return euclidean_domain_gurobi(
        [[random(), random(), random(), random(), random()]
         for _ in range(num_candidates)], bounding_box=bounding_box)


@validate_domain
def ext_euclidean_1d_domain(num_candidates):
    domain = euclidean_1d_domain(num_candidates)
    return extend_to_reversed(domain)

@validate_domain
def ext_euclidean_2d_domain(num_candidates):
    domain = euclidean_2d_domain(num_candidates)
    return extend_to_reversed(domain)

@validate_domain
def ext_euclidean_3d_domain(num_candidates):
    domain = euclidean_3d_domain(num_candidates)
    return extend_to_reversed(domain)