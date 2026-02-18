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


class GearSubproblem:
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
        consis_vio = self._target_shared_var - x[self._aux_idx]
        penalty_term = (
            self._lagrange_est @ consis_vio
            + (np.linalg.norm(self._penalty_weight * consis_vio, ord=2)) ** 2
        )

        x_0 = x[self._local_idx.index(0)]
        x_1 = x[self._local_idx.index(1)]
        x_2 = x[self._local_idx.index(2)]

        f_1 = 0.7854 * x_0 * x_1**2 * (3.3333 * x_2**2 + 14.9335 * x_2 - 43.0934)
        obj = f_1 + penalty_term

        g_5 = 27 / (x_0 * x_1**2 * x_2) - 1
        g_6 = 397.5 / (x_0 * x_1**2 * x_2**2) - 1
        g_9 = (x_1 * x_2) / 40 - 1
        g_10 = (5 * x_1) / x_0 - 1
        g_11 = x_0 / (12 * x_1) - 1
        return [obj, g_5, g_6, g_9, g_10, g_11]

    def get_bounds(self):
        lb_local = [lb[idx] for idx in self._local_idx]
        ub_local = [ub[idx] for idx in self._local_idx]
        return (lb_local, ub_local)

    def get_nic(self):
        return 5

    def get_nec(self):
        return 0

    def gradient(self, x):
        return pg.estimate_gradient_h(lambda x: self.fitness(x), x)


class Shaft1Subproblem:
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
        consis_vio = self._target_shared_var - x[self._aux_idx]
        penalty_term = (
            self._lagrange_est @ consis_vio
            + (np.linalg.norm(self._penalty_weight * consis_vio, ord=2)) ** 2
        )

        x_0 = x[self._local_idx.index(0)]
        x_1 = x[self._local_idx.index(1)]
        x_2 = x[self._local_idx.index(2)]
        x_3 = x[self._local_idx.index(3)]
        x_5 = x[self._local_idx.index(5)]

        f_2 = -1.5079 * x_0 * x_5**2
        f_4 = 7.477 * x_5**3
        f_6 = 0.7854 * x_3 * x_5**2
        obj = f_2 + f_4 + f_6 + penalty_term

        g_1 = (1 / (110 * x_5**3)) * np.sqrt(
            (745 * x_3 / (x_1 * x_2)) ** 2 + 1.69e7
        ) - 1
        g_3 = (1.5 * x_5 + 1.9) / x_3 - 1
        g_7 = (1.93 * x_3**3) / (x_1 * x_2 * x_5**4) - 1

        return [obj, g_1, g_3, g_7]

    def get_bounds(self):
        lb_local = [lb[idx] for idx in self._local_idx]
        ub_local = [ub[idx] for idx in self._local_idx]
        return (lb_local, ub_local)

    def get_nic(self):
        return 3

    def get_nec(self):
        return 0

    def gradient(self, x):
        return pg.estimate_gradient_h(lambda x: self.fitness(x), x)


class Shaft2Subproblem:
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
        consis_vio = self._target_shared_var - x[self._aux_idx]
        penalty_term = (
            self._lagrange_est @ consis_vio
            + (np.linalg.norm(self._penalty_weight * consis_vio, ord=2)) ** 2
        )

        x_0 = x[self._local_idx.index(0)]
        x_1 = x[self._local_idx.index(1)]
        x_2 = x[self._local_idx.index(2)]
        x_4 = x[self._local_idx.index(4)]
        x_6 = x[self._local_idx.index(6)]

        f_3 = -1.5079 * x_0 * x_6**2
        f_5 = 7.477 * x_6**3
        f_7 = 0.7854 * x_4 * x_6**2
        obj = f_3 + f_5 + f_7 + penalty_term

        g_2 = (1 / (85 * x_6**3)) * np.sqrt(
            (745 * x_4 / (x_1 * x_2)) ** 2 + 1.575e8
        ) - 1
        g_4 = (1.1 * x_6 + 1.9) / x_4 - 1
        g_8 = (1.93 * x_4**3) / (x_1 * x_2 * x_6**4) - 1
        return [obj, g_2, g_4, g_8]

    def get_bounds(self):
        lb_local = [lb[idx] for idx in self._local_idx]
        ub_local = [ub[idx] for idx in self._local_idx]
        return (lb_local, ub_local)

    def get_nic(self):
        return 3

    def get_nec(self):
        return 0

    def gradient(self, x):
        return pg.estimate_gradient_h(lambda x: self.fitness(x), x)


all_subprob_dict = {
    0: {
        "optim type": "NLP",
        "function": solve_nlp_subp_w_pygmo,
        "args": GearSubproblem,
    },
    1: {
        "optim type": "NLP",
        "function": solve_nlp_subp_w_pygmo,
        "args": Shaft1Subproblem,
    },
    2: {
        "optim type": "NLP",
        "function": solve_nlp_subp_w_pygmo,
        "args": Shaft2Subproblem,
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
