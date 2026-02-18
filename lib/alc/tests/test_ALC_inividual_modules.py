"""
ALC test problem: Golinski's speed reducer
For details, see http://dx.doi.org/10.1002/nme.2158
"""

import pytest
import numpy as np

try:
    # if used as a subtree
    from lib.alc.alc import (
        AllSubpDict,
    )
except (ModuleNotFoundError, ImportError):
    # for stant-alone testing
    from alc import (
        AllSubpDict,
    )

try:
    # if used as a subtree
    from lib.alc.alc import InnerLoop, OuterLoop, DimensionConverter
except (ModuleNotFoundError, ImportError):
    # for stant-alone testing
    from alc import InnerLoop, OuterLoop, DimensionConverter


initial_guess: np.ndarray = np.array(
    [2.6168, 0.7440, 18.6323, 7.9214, 8.0544, 3.1806, 5.2636]
)
initial_guess: np.ndarray = np.array(
    [3.4980, 0.7029, 20.1022, 7.4879, 7.9088, 3.1094, 5.4989]
)
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


def _noop(*args, **kwargs):
    raise AssertionError("This function should not be called in the test.")


all_subprob_dict: AllSubpDict = {
    0: {"optim type": "NLP", "function": _noop, "args": None},
    1: {"optim type": "NLP", "function": _noop, "args": None},
    2: {"optim type": "NLP", "function": _noop, "args": None},
}
dc = DimensionConverter(
    dependency_matrix=depen_matrix,
    dim_all_var=dim_all_var,
)


@pytest.fixture(scope="module")
def inner_loop_inst():
    inner_loop = InnerLoop(
        dc=dc,
        all_subprob_dict=all_subprob_dict,
        initial_guess=initial_guess,
        tol_inner=1e-5,
    )
    inner_loop.lagrange_est_list = [
        0.01 + np.zeros(dim) for dim in dc.dim_aux_shared_var
    ]
    inner_loop.penalty_weight_list = [
        0.1 + np.zeros(dim) for dim in dc.dim_aux_shared_var
    ]
    inner_loop.outer_counter = 1
    return inner_loop


@pytest.fixture(scope="module")
def outer_loop_inst(inner_loop_inst):
    outer_loop = OuterLoop(
        inner_loop=inner_loop_inst,
        dc=dc,
        initial_weight=0.001,
        initial_weight_coefficient=0.01,
        weight_update_coefficient=1.5,
        weight_update_fraction=0.5,
        tol_outer=1e-3,
        update_initial_weight=False,
    )
    return outer_loop


def test_solve_master_prob(inner_loop_inst):
    shared_var = inner_loop_inst._solve_master_problem(
        [
            np.array([2.6, 0.7, 17]),
            np.array([3.6, 0.8, 28]),
            np.array([3.0, 0.72, 20]),
        ]
    )
    np.testing.assert_array_almost_equal(
        shared_var,
        np.array(
            [
                (2.6 + 3.6 + 3.0) / 3 - 0.5,
                (0.7 + 0.8 + 0.72) / 3 - 0.5,
                (17 + 28 + 20) / 3 - 0.5,
            ]
        ),
        decimal=2,
    )


def test_update_initial_guess(inner_loop_inst):
    shared_var = np.array([3.0, 0.75, 20])
    local_design_var = [
        np.array([2.6, 0.7, 17]),
        np.array([3.6, 0.8, 28, 8.0, 3.0]),
        np.array([3.0, 0.72, 20, 7.3, 5.0]),
    ]
    initial_guess = inner_loop_inst._update_initial_guess(
        shared_var=shared_var,
        local_design_var=local_design_var,
    )
    np.testing.assert_array_almost_equal(
        initial_guess,
        np.array([3.0, 0.75, 20, 8.0, 7.3, 3.0, 5.0]),
        decimal=2,
    )


def test_update_penalty_parameters(outer_loop_inst):
    updated_params = outer_loop_inst._update_penalty_parameters(
        lagrange_est=np.array([1.0, 2.0, 3.0]),
        penalty_weight=np.array([3.0, 2.0, 1.0]),
        consis_vio=np.array([0.01, -0.1, 0.2]),
        consis_vio_prev=np.array([0.1, 0.1, 0.1]),
        update_weight=True,
    )
    np.testing.assert_array_almost_equal(
        updated_params["lagrange est"],
        np.array([1.18, 1.2, 3.4]),
        decimal=2,
    )
    np.testing.assert_array_almost_equal(
        updated_params["penalty weight"],
        np.array([3.0, 3.0, 1.5]),
        decimal=2,
    )
