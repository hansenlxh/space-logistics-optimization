"""
ALC test problem: Golinski's speed reducer
For details, see http://dx.doi.org/10.1002/nme.2158
"""

import pytest
import numpy as np

pg = pytest.importorskip("pygmo")
pytestmark = pytest.mark.legacy

try:
    # if used as a subtree
    from lib.alc.alc import (
        InnerLoop,
        OuterLoop,
        DimensionConverter,
    )
    from lib.alc.alc.legacy.subproblems_pygmo import (
        solve_subp_pygmo_wrapper,
        solve_nlp_subp_w_pygmo,
    )
except (ModuleNotFoundError, ImportError):
    # for stant-alone testing
    from alc import (
        InnerLoop,
        OuterLoop,
        DimensionConverter,
    )
    from alc.legacy.subproblems_pygmo import (
        solve_subp_pygmo_wrapper,
        solve_nlp_subp_w_pygmo,
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


class Problem1:
    def __init__(
        self,
        target_shared_var: np.ndarray,
        lagrange_est: np.ndarray,
        penalty_weight: np.ndarray,
        local_var_idx: list[int],
        aux_shared_var_idx: list[int],
        initial_guess: np.ndarray | None = None,
        args=None,
    ):
        self._target_shared_var = target_shared_var
        self._lagrange_est = lagrange_est
        self._penalty_weight = penalty_weight
        self._aux_idx = aux_shared_var_idx
        self._local_idx = local_var_idx
        self._args = args

    def fitness(self, x):
        local_shared_idx = [self._local_idx.index(idx) for idx in self._aux_idx]
        consis_vio = self._target_shared_var - x[local_shared_idx]
        penalty_term = (
            self._lagrange_est @ consis_vio
            + (np.linalg.norm(self._penalty_weight * consis_vio, ord=2)) ** 2
        )

        x_0 = x[self._local_idx.index(0)]
        x_1 = x[self._local_idx.index(1)]
        x_2 = x[self._local_idx.index(2)]
        x_3 = x[self._local_idx.index(3)]
        x_4 = x[self._local_idx.index(4)]
        x_5 = x[self._local_idx.index(5)]
        x_6 = x[self._local_idx.index(6)]

        obj = x_0**2 + x_1**2 + penalty_term

        h1 = (x_2**2 + x_3 ** (-2) + x_4**2) * (x_0**-2) - 1
        h2 = (x_4**2 + x_5**2 + x_6**2) * (x_1**-2) - 1
        g1 = (x_2**-2 + x_3**2) * (x_4**-2) - 1
        g2 = (x_4**2 + x_5**-2) * (x_6**-2) - 1

        return [obj, h1, h2, g1, g2]

    def get_bounds(self):
        lb_local = [lb[idx] for idx in self._local_idx]
        ub_local = [ub[idx] for idx in self._local_idx]
        return (lb_local, ub_local)

    def get_nic(self):
        return 2

    def get_nec(self):
        return 2

    def gradient(self, x):
        return pg.estimate_gradient_h(lambda x: self.fitness(x), x)


class Problem2:
    def __init__(
        self,
        target_shared_var: np.ndarray,
        lagrange_est: np.ndarray,
        penalty_weight: np.ndarray,
        local_var_idx: list[int],
        aux_shared_var_idx: list[int],
        initial_guess: np.ndarray | None = None,
        args=None,
    ):
        self._target_shared_var = target_shared_var
        self._lagrange_est = lagrange_est
        self._penalty_weight = penalty_weight
        self._aux_idx = aux_shared_var_idx
        self._local_idx = local_var_idx
        self._args = args

    def fitness(self, x):
        local_shared_idx = [self._local_idx.index(idx) for idx in self._aux_idx]
        consis_vio = self._target_shared_var - x[local_shared_idx]
        penalty_term = (
            self._lagrange_est @ consis_vio
            + (np.linalg.norm(self._penalty_weight * consis_vio, ord=2)) ** 2
        )

        x_2 = x[self._local_idx.index(2)]
        x_7 = x[self._local_idx.index(7)]
        x_8 = x[self._local_idx.index(8)]
        x_9 = x[self._local_idx.index(9)]
        x_10 = x[self._local_idx.index(10)]

        obj = penalty_term

        h3 = (x_7**2 + x_8**-2 + x_9**-2 + x_10**2) * (x_2**-2) - 1
        g3 = (x_7**2 + x_8**2) * (x_10**-2) - 1
        g4 = (x_7**-2 + x_9**2) * (x_10**-2) - 1

        return [obj, h3, g3, g4]

    def get_bounds(self):
        lb_local = [lb[idx] for idx in self._local_idx]
        ub_local = [ub[idx] for idx in self._local_idx]
        return (lb_local, ub_local)

    def get_nic(self):
        return 2

    def get_nec(self):
        return 1

    def gradient(self, x):
        return pg.estimate_gradient_h(lambda x: self.fitness(x), x)


class Problem3:
    def __init__(
        self,
        target_shared_var: np.ndarray,
        lagrange_est: np.ndarray,
        penalty_weight: np.ndarray,
        local_var_idx: list[int],
        aux_shared_var_idx: list[int],
        initial_guess: np.ndarray | None = None,
        args=None,
    ):
        self._target_shared_var = target_shared_var
        self._lagrange_est = lagrange_est
        self._penalty_weight = penalty_weight
        self._aux_idx = aux_shared_var_idx
        self._local_idx = local_var_idx
        self._args = args

    def fitness(self, x):
        local_shared_idx = [self._local_idx.index(idx) for idx in self._aux_idx]
        consis_vio = self._target_shared_var - x[local_shared_idx]
        penalty_term = (
            self._lagrange_est @ consis_vio
            + (np.linalg.norm(self._penalty_weight * consis_vio, ord=2)) ** 2
        )

        x_5 = x[self._local_idx.index(5)]
        x_10 = x[self._local_idx.index(10)]
        x_11 = x[self._local_idx.index(11)]
        x_12 = x[self._local_idx.index(12)]
        x_13 = x[self._local_idx.index(13)]

        obj = penalty_term

        h4 = (x_10**2 + x_11**2 + x_12**2 + x_13**2) * (x_5**-2) - 1
        g5 = (x_10**2 + x_11**-2) * (x_12**-2) - 1
        g6 = (x_10**2 + x_11**2) * (x_13**-2) - 1
        return [obj, h4, g5, g6]

    def get_bounds(self):
        lb_local = [lb[idx] for idx in self._local_idx]
        ub_local = [ub[idx] for idx in self._local_idx]
        return (lb_local, ub_local)

    def get_nic(self):
        return 2

    def get_nec(self):
        return 1

    def gradient(self, x):
        return pg.estimate_gradient_h(lambda x: self.fitness(x), x)


all_subprob_dict = {
    0: {
        "optim type": "NLP",
        "function": solve_nlp_subp_w_pygmo,
        "args": Problem1,
    },
    1: {
        "optim type": "NLP",
        "function": solve_nlp_subp_w_pygmo,
        "args": Problem2,
    },
    2: {
        "optim type": "NLP",
        "function": solve_nlp_subp_w_pygmo,
        "args": Problem3,
    },
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
    solve_fn=solve_subp_pygmo_wrapper,  # must be specified for pygmo
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
    inner_loop=inner_loop,
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
