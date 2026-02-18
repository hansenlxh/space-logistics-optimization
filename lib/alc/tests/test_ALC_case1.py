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
    [2.6168, 0.7440, 18.6323, 7.9214, 8.0544, 3.1806, 5.2636]
)
# initial_guess: np.ndarray = np.array(
#     [3.4980, 0.7029, 20.1022, 7.4879, 7.9088, 3.1094, 5.4989]
# )
lb = [2.6, 0.7, 17, 7.3, 7.3, 2.9, 5.0]
ub = [3.6, 0.8, 28, 8.3, 8.3, 3.9, 5.5]
depen_matrix: list[list[int]] = [
    [1, 1, 1, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 1, 0],
    [1, 1, 1, 0, 1, 0, 1],
]
dim_all_var = 7
n_subprob = 3
known_solution = np.array([3.50, 0.70, 17.00, 7.30, 7.72, 3.35, 5.29])
known_obj = 2994


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


def solve_gear_subproblem(
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

    m.const_g5 = Constraint(expr=27 / (x0 * x1**2 * x2) <= 1)
    m.const_g6 = Constraint(expr=397.5 / (x0 * x1**2 * x2**2) <= 1)
    m.const_g9 = Constraint(expr=(x1 * x2) / 40 <= 1)
    m.const_g10 = Constraint(expr=(5 * x1) / x0 <= 1)
    m.const_g11 = Constraint(expr=x0 / (12 * x1) <= 1)

    m.obj = Objective(
        expr=0.7854 * x0 * x1**2 * (3.3333 * x2**2 + 14.9335 * x2 - 43.0934)
        + penalty_term,
        sense=pyo.minimize,
    )

    SolverFactory("ipopt").solve(m, tee=False)
    obj = float(pyo.value(m.obj))
    design_var = np.array([pyo.value(m.x[i]) for i in m.idx], dtype=float)

    subp_res: SubproblemResult = {"objective": obj, "design var": design_var}
    return subp_res


def solve_shaft1_subproblem(
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
    x5 = m.x[local_var_idx.index(5)]

    m.const_g1 = Constraint(
        expr=(1 / (110 * x5**3)) * pyo.sqrt((745 * x3 / (x1 * x2)) ** 2 + 1.69e7) <= 1
    )
    m.const_g3 = Constraint(expr=(1.5 * x5 + 1.9) / x3 <= 1)
    m.const_g7 = Constraint(expr=(1.93 * x3**3) / (x1 * x2 * x5**4) <= 1)

    f2 = -1.5079 * x0 * x5**2
    f4 = 7.477 * x5**3
    f6 = 0.7854 * x3 * x5**2
    m.obj = Objective(
        expr=f2 + f4 + f6 + penalty_term,
        sense=pyo.minimize,
    )

    SolverFactory("ipopt").solve(m, tee=False)
    obj = float(pyo.value(m.obj))
    design_var = np.array([pyo.value(m.x[i]) for i in m.idx], dtype=float)

    subp_res: SubproblemResult = {"objective": obj, "design var": design_var}
    return subp_res


def solve_shaft2_subproblem(
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
    x4 = m.x[local_var_idx.index(4)]
    x6 = m.x[local_var_idx.index(6)]

    m.const_g2 = Constraint(
        expr=(1 / (85 * x6**3)) * pyo.sqrt((745 * x4 / (x1 * x2)) ** 2 + 1.575e8) <= 1
    )
    m.const_g4 = Constraint(expr=(1.1 * x6 + 1.9) / x4 <= 1)
    m.const_g8 = Constraint(expr=(1.93 * x4**3) / (x1 * x2 * x6**4) <= 1)

    f3 = -1.5079 * x0 * x6**2
    f5 = 7.477 * x6**3
    f7 = 0.7854 * x4 * x6**2
    m.obj = Objective(
        expr=f3 + f5 + f7 + penalty_term,
        sense=pyo.minimize,
    )

    SolverFactory("ipopt").solve(m, tee=False)
    obj = float(pyo.value(m.obj))
    design_var = np.array([pyo.value(m.x[i]) for i in m.idx], dtype=float)

    subp_res: SubproblemResult = {"objective": obj, "design var": design_var}
    return subp_res


all_subprob_dict: AllSubpDict = {
    0: {"optim type": "NLP", "function": solve_gear_subproblem, "args": None},
    1: {"optim type": "NLP", "function": solve_shaft1_subproblem, "args": None},
    2: {"optim type": "NLP", "function": solve_shaft2_subproblem, "args": None},
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
    initial_weight=0.001,
    initial_weight_coefficient=0.01,
    weight_update_coefficient=1.5,
    weight_update_fraction=0.5,
    tol_outer=1e-3,
    update_initial_weight=False,
)


def test_outer_loop():
    result = outer_loop.run()
    assert sum(result["objectives"]) == pytest.approx(known_obj, 1e-2)
    np.testing.assert_array_almost_equal(
        result["design vars"],
        known_solution,
        decimal=2,
    )


inner_loop_admm = InnerLoop(
    dc=dc,
    all_subprob_dict=all_subprob_dict,
    initial_guess=initial_guess,
    use_admm=True,
    tol_inner=1e-5,
)
outer_loop_admm = OuterLoop(
    inner_loop=inner_loop_admm,
    dc=dc,
    initial_weight=0.001,
    initial_weight_coefficient=0.01,
    weight_update_coefficient=1.2,
    weight_update_fraction=0.8,
    tol_outer=1e-3,
    update_initial_weight=False,
)


def test_outer_loop_admm():
    result = outer_loop_admm.run()
    assert sum(result["objectives"]) == pytest.approx(known_obj, 1e-2)
    np.testing.assert_array_almost_equal(
        result["design vars"],
        known_solution,
        decimal=2,
    )
