"""
ALC test problem: Golinski's speed reducer
For details, see http://dx.doi.org/10.1002/nme.2158
"""

import pytest
import numpy as np
from pyomo.environ import (
    ConcreteModel,
    RangeSet,
    Var,
    Objective,
    Constraint,
    SolverFactory,
    Reals,
)
import pyomo.environ as pyo

try:
    # if used as a subtree
    from lib.alc.alc import (
        InnerLoop,
        OuterLoop,
        DimensionConverter,
        SubproblemResult,
        AllSubpDict,
    )
except (ModuleNotFoundError, ImportError):
    # for stant-alone testing
    from alc import (
        InnerLoop,
        OuterLoop,
        DimensionConverter,
        SubproblemResult,
        AllSubpDict,
    )


initial_guess: np.ndarray = np.array(
    [
        0.7781,
        4.0797,
        4.5080,
        3.2505,
        3.1260,
        3.7683,
        3.1040,
        3.1589,
        3.2777,
        2.0681,
        2.9733,
        2.4467,
        3.5178,
        0.7002,
    ]
)
dim_all_var = len(initial_guess)
lb = [0] * dim_all_var
ub = [10] * dim_all_var
depen_matrix: list[list[int]] = [
    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1],
]
n_subprob = 3
known_solution = np.array(
    [2.84, 3.09, 2.36, 0.76, 0.87, 2.81, 0.94, 0.97, 0.87, 0.80, 1.30, 0.84, 1.76, 1.55]
)
known_obj = 17.59


def _build_base_model(
    target_shared_var: np.ndarray,
    lagrange_est: np.ndarray,
    penalty_weight: np.ndarray,
    local_var_idx: list[int],
    aux_shared_var_idx: list[int],
    initial_guess=initial_guess,  # in local index
):
    dim_local_var = len(local_var_idx)
    dim_aux_shared_var = len(aux_shared_var_idx)
    m = ConcreteModel()
    m.idx = RangeSet(0, dim_local_var - 1)
    m.shared_idx = RangeSet(0, dim_aux_shared_var - 1)
    m.x = Var(m.idx, domain=Reals)

    for i in range(dim_local_var):
        m.x[i].value = initial_guess[i]
        m.x[i].lb = lb[local_var_idx[i]]
        m.x[i].ub = ub[local_var_idx[i]]

    aux_local_pos = [local_var_idx.index(gid) for gid in aux_shared_var_idx]
    penalty_term = sum(
        float(lagrange_est[k]) * (float(target_shared_var[k]) - m.x[aux_local_pos[k]])
        + (
            float(penalty_weight[k])
            * (float(target_shared_var[k]) - m.x[aux_local_pos[k]])
        )
        ** 2
        for k in range(dim_aux_shared_var)
    )

    return m, penalty_term


def solve_subproblem1(
    target_shared_var: np.ndarray,
    lagrange_est: np.ndarray,
    penalty_weight: np.ndarray,
    local_var_idx: list[int],
    aux_shared_var_idx: list[int],
    args=None,
    initial_guess=initial_guess,  # in local index
) -> SubproblemResult:
    m, penalty_term = _build_base_model(
        target_shared_var,
        lagrange_est,
        penalty_weight,
        local_var_idx,
        aux_shared_var_idx,
        initial_guess,
    )
    x0 = m.x[local_var_idx.index(0)]
    x1 = m.x[local_var_idx.index(1)]
    x2 = m.x[local_var_idx.index(2)]
    x3 = m.x[local_var_idx.index(3)]
    x4 = m.x[local_var_idx.index(4)]
    x5 = m.x[local_var_idx.index(5)]
    x6 = m.x[local_var_idx.index(6)]

    m.const_h1 = Constraint(expr=(x2**2 + x3 ** (-2) + x4**2) * (x0**-2) == 1)
    m.const_h2 = Constraint(expr=(x4**2 + x5**2 + x6**2) * (x1**-2) == 1)
    m.const_g1 = Constraint(expr=(x2**-2 + x3**2) * (x4**-2) <= 1)
    m.const_g2 = Constraint(expr=(x4**2 + x5**-2) * (x6**-2) <= 1)

    m.obj = Objective(
        expr=x0**2 + x1**2 + penalty_term,
        sense=pyo.minimize,
    )

    SolverFactory("ipopt").solve(m, tee=False)
    obj = float(pyo.value(m.obj))
    design_var = np.array([pyo.value(m.x[i]) for i in m.idx], dtype=float)

    subp_res: SubproblemResult = {"objective": obj, "design var": design_var}
    return subp_res


def solve_subproblem2(
    target_shared_var: np.ndarray,
    lagrange_est: np.ndarray,
    penalty_weight: np.ndarray,
    local_var_idx: list[int],
    aux_shared_var_idx: list[int],
    args=None,
    initial_guess=initial_guess,  # in local index
) -> SubproblemResult:
    m, penalty_term = _build_base_model(
        target_shared_var,
        lagrange_est,
        penalty_weight,
        local_var_idx,
        aux_shared_var_idx,
        initial_guess,
    )
    x2 = m.x[local_var_idx.index(2)]
    x7 = m.x[local_var_idx.index(7)]
    x8 = m.x[local_var_idx.index(8)]
    x9 = m.x[local_var_idx.index(9)]
    x10 = m.x[local_var_idx.index(10)]

    m.const_h3 = Constraint(expr=(x7**2 + x8**-2 + x9**-2 + x10**2) * (x2**-2) == 1)
    m.const_g3 = Constraint(expr=(x7**2 + x8**2) * (x10**-2) <= 1)
    m.const_g4 = Constraint(expr=(x7**-2 + x9**2) * (x10**-2) <= 1)

    m.obj = Objective(
        expr=penalty_term,
        sense=pyo.minimize,
    )

    SolverFactory("ipopt").solve(m, tee=False)
    obj = float(pyo.value(m.obj))
    design_var = np.array([pyo.value(m.x[i]) for i in m.idx], dtype=float)

    subp_res: SubproblemResult = {"objective": obj, "design var": design_var}
    return subp_res


def solve_subproblem3(
    target_shared_var: np.ndarray,
    lagrange_est: np.ndarray,
    penalty_weight: np.ndarray,
    local_var_idx: list[int],
    aux_shared_var_idx: list[int],
    args=None,
    initial_guess=initial_guess,  # in local index
) -> SubproblemResult:
    m, penalty_term = _build_base_model(
        target_shared_var,
        lagrange_est,
        penalty_weight,
        local_var_idx,
        aux_shared_var_idx,
        initial_guess,
    )
    x5 = m.x[local_var_idx.index(5)]
    x10 = m.x[local_var_idx.index(10)]
    x11 = m.x[local_var_idx.index(11)]
    x12 = m.x[local_var_idx.index(12)]
    x13 = m.x[local_var_idx.index(13)]

    m.const_h4 = Constraint(expr=(x10**2 + x11**2 + x12**2 + x13**2) * (x5**-2) == 1)
    m.const_g5 = Constraint(expr=(x10**2 + x11**-2) * (x12**-2) <= 1)
    m.const_g6 = Constraint(expr=(x10**2 + x11**2) * (x13**-2) <= 1)

    m.obj = Objective(
        expr=penalty_term,
        sense=pyo.minimize,
    )

    solver = SolverFactory("ipopt")
    solver.options["tol"] = 1e-8
    solver.options["constr_viol_tol"] = 1e-8
    solver.solve(m, tee=False)

    obj = float(pyo.value(m.obj))
    design_var = np.array([pyo.value(m.x[i]) for i in m.idx], dtype=float)

    subp_res: SubproblemResult = {"objective": obj, "design var": design_var}
    return subp_res


all_subprob_dict: AllSubpDict = {
    0: {"optim type": "NLP", "function": solve_subproblem1, "args": None},
    1: {"optim type": "NLP", "function": solve_subproblem2, "args": None},
    2: {"optim type": "NLP", "function": solve_subproblem3, "args": None},
}
dc = DimensionConverter(
    dependency_matrix=depen_matrix,
    dim_all_var=dim_all_var,
)
inner_loop = InnerLoop(
    dc=dc,
    all_subprob_dict=all_subprob_dict,
    initial_guess=initial_guess,
    tol_inner=1e-5,
)
outer_loop = OuterLoop(
    inner_loop=inner_loop,
    dc=dc,
    initial_weight=1,
    initial_weight_coefficient=1,
    weight_update_coefficient=1.5,
    weight_update_fraction=0.5,
    tol_outer=1e-3,
    update_initial_weight=True,
)


def test_outer_loop():
    result = outer_loop.run()
    assert sum(result["objectives"]) == pytest.approx(known_obj, 1e-2)
    np.testing.assert_allclose(
        result["design vars"],
        known_solution,
        rtol=0.1,
    )


inner_loop_admm = InnerLoop(
    dc=dc,
    all_subprob_dict=all_subprob_dict,
    initial_guess=initial_guess,
    use_admm=True,
    tol_inner=1e-5,
)
outer_loop_admm = OuterLoop(
    inner_loop=inner_loop,  # _admm
    dc=dc,
    initial_weight=1,
    initial_weight_coefficient=1,
    weight_update_coefficient=1.2,
    weight_update_fraction=0.8,
    tol_outer=1e-3,
    update_initial_weight=True,
)


def test_outer_loop_admm():
    result = outer_loop_admm.run()
    assert sum(result["objectives"]) == pytest.approx(known_obj, 1e-2)
    np.testing.assert_allclose(
        result["design vars"],
        known_solution,
        rtol=0.1,
    )
