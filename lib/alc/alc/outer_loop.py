import numpy as np
import multiprocessing as mP
import pandas as pd
import time
from pandas import DataFrame, Series
from dataclasses import dataclass
from icecream import ic
from .inner_loop import InnerLoop
from .dimension_converter import DimensionConverter


class OuterLoop:
    def __init__(
        self,
        inner_loop: InnerLoop,
        dc: DimensionConverter,
        initial_weight: float,
        initial_weight_coefficient: float,
        weight_update_coefficient: float,
        weight_update_fraction: float,
        tol_outer: float = 1e-3,
        update_initial_weight: bool = False,
        verbose: bool = False,
        store_results: bool = False,
    ) -> None:
        """Constructor of the outer loop of the ALC algorithm

        Args:
            inner_loop: instance of ALC InnerLoop class
            dc: instance of DimensionConverter class
            initial_weight: initial penalty weight
            initial_weight_coefficient: coefficient for re-calculating
                initial penalty weight based on the first inner loop iteration
            weight_update_coefficient: coefficient for updating penalty weight
                higher value means faster increase of penalty weight
            weight_update_fraction: fraction of consistency violation
                to determine if penalty weight should be increased or not
                smaller value means more frequent increase of penalty weight
            tol_outer: tolerance for the outer loop convergence
            update_initial_weight: indicator to update initial weight
                based on the first inner loop iteration
                Defaults to False
            verbose: indicator to print iteration information, defaults to False
            store_results: indicator to store iteration logs (outer loop only)
                To access the logs, use the attribute `iter_logs`
        """
        self._inner_loop = inner_loop
        self._dc = dc
        self._initial_weight = initial_weight
        self._initial_weight_coefficient = initial_weight_coefficient
        self._weight_update_coefficient = weight_update_coefficient
        self._weight_update_fraction = weight_update_fraction
        self._tol_outer = tol_outer
        self._update_initial_weight = update_initial_weight
        self._n_subprob = self._dc.n_subprob
        self._verbose = verbose
        self._store_results = store_results
        self.iter_logs: DataFrame | None = None

    def run(self) -> dict[str, list[float] | np.ndarray]:
        """Run ALC outer loop

        Returns:
            Dictionary containing outer loop results
                "objectives": list of objective values
                "desing vars": desing variables
        """
        # initialize Lagrange multiplier estimate and penalty weight
        lagrange_est_list: list[np.ndarray] = [
            np.zeros(dim) for dim in self._dc.dim_aux_shared_var
        ]
        penalty_weight_list: list[np.ndarray] = [
            self._initial_weight + np.zeros(dim) for dim in self._dc.dim_aux_shared_var
        ]

        # place hoders
        consis_vio_list_curr: list[np.ndarray] = [0] * self._n_subprob
        consis_vio_list_prev: list[np.ndarray] = [0] * self._n_subprob
        vio_abs_list: list[np.ndarray] = [0] * self._n_subprob
        vio_imprv_list: list[np.ndarray] = [0] * self._n_subprob

        outer_counter = 0

        if self._store_results:
            iteration_records = []
        start_time = time.time()

        while (
            outer_counter <= 1
            or any(np.concatenate(vio_abs_list) > self._tol_outer)
            or any(np.concatenate(vio_imprv_list) > self._tol_outer)
        ):
            outer_counter += 1
            if self._verbose:
                print("\n###### Outer Loop Iteration: ", outer_counter)
            if outer_counter > 1:
                consis_vio_list_prev = consis_vio_list_curr

            """ run inner loop """
            self._inner_loop.lagrange_est_list = lagrange_est_list
            self._inner_loop.penalty_weight_list = penalty_weight_list
            self._inner_loop.outer_counter = outer_counter
            inner_result: dict = self._inner_loop.run()
            shared_var: np.ndarray = inner_result["shared"]
            aux_shared_var: list[np.ndarray] = inner_result["aux shared"]
            curr_sol: np.ndarray = inner_result["all"]
            obj_list: list[float] = inner_result["objectives"]
            target_var_list: list[np.ndarray] = [
                self._dc.shared_to_aux_shared(shared_var, sp_id)
                for sp_id in range(self._n_subprob)
            ]

            """ Update Lagrange multiplier estimate and penalty weight """
            for sp_id in range(self._n_subprob):
                consis_vio: np.ndarray = target_var_list[sp_id] - aux_shared_var[sp_id]
                consis_vio_list_curr[sp_id] = consis_vio

            if self._update_initial_weight and outer_counter == 1:
                penalty_weight_list = self._update_initial_weight_list(
                    obj_list, consis_vio_list_curr
                )

            for sp_id in range(self._n_subprob):
                updated_params = self._update_penalty_parameters(
                    lagrange_est=lagrange_est_list[sp_id],
                    penalty_weight=penalty_weight_list[sp_id],
                    consis_vio=consis_vio_list_curr[sp_id],
                    consis_vio_prev=consis_vio_list_prev[sp_id],
                    update_weight=(outer_counter > 1),
                )
                lagrange_est_list[sp_id] = updated_params["lagrange est"]
                penalty_weight_list[sp_id] = updated_params["penalty weight"]

                # calculate metrics for convergence check
                vio_imprv_list[sp_id] = abs(
                    consis_vio_list_curr[sp_id] - consis_vio_list_prev[sp_id]
                ) / (1 + abs(target_var_list[sp_id]))
                vio_abs_list[sp_id] = abs(consis_vio_list_curr[sp_id]) / (
                    1 + abs(target_var_list[sp_id])
                )

                if self._verbose:
                    ic(sp_id)
                    ic(consis_vio_list_curr[sp_id])
                    ic(lagrange_est_list[sp_id])
                    ic(penalty_weight_list[sp_id])

            if self._store_results:
                elapsed_time = time.time() - start_time
                record = {
                    "Outer Iteration": outer_counter,
                    "Elapsed Time (s)": elapsed_time,
                    "Max Abs. Violation": max(
                        np.linalg.norm(v, ord=np.inf) for v in vio_abs_list
                    ),
                    "Max Violation Change": max(
                        np.linalg.norm(v, ord=np.inf) for v in vio_imprv_list
                    ),
                    "Objectives": obj_list.copy(),
                    "Design Vars": curr_sol.copy(),
                }
                iteration_records.append(record)

        if self._store_results:
            self.iter_logs = pd.DataFrame(iteration_records)
        outer_loop_results: dict = {
            "objectives": obj_list,
            "design vars": curr_sol,
        }
        return outer_loop_results

    def _update_initial_weight_list(
        self, obj_list: list[float], consis_vio_list_curr: list[np.ndarray]
    ) -> list[np.ndarray]:
        """Update initial penalty weight after the first inner loop convergence

        Args:
            obj_list: list of objective values
            consis_vio_list_curr: list of consistency violation values
        Returns:
            penalty_weight_list: list of penalty weights for each subproblem
        """
        flattened_vio_array: np.ndarray = np.concatenate(
            [array.ravel() for array in consis_vio_list_curr]
        )
        self._initial_weight = np.sqrt(
            self._initial_weight_coefficient
            * abs(sum(obj_list))
            / max((np.linalg.norm(flattened_vio_array) ** 2, 1))
        )
        penalty_weight_list = [
            self._initial_weight + np.zeros(dim) for dim in self._dc.dim_aux_shared_var
        ]
        return penalty_weight_list

    def _update_penalty_parameters(
        self,
        lagrange_est: np.ndarray,
        penalty_weight: np.ndarray,
        consis_vio: np.ndarray,
        consis_vio_prev: np.ndarray,
        update_weight: bool,
    ) -> dict[str, np.ndarray]:
        """Update Lagrange multiplier estimate and penalty weights
        for a single subproblem

        Args:
            lagrange_est: current Lagrange multiplier estimate
            penalty_weight: current penalty weight
            consis_vio: current consistency violation
            consis_vio_prev: previous consistency violation
            update_weight: indicator to update penalty weight or not
                True except for the first outer loop iteration
        Returns:
            Dictionary containing updated Lagrange multiplier estimate and penalty weight
                "lagrange est": updated Lagrange multiplier estimate
                "penalty weight": updated penalty weight
        """

        lagrange_est += 2 * penalty_weight * penalty_weight * consis_vio

        if update_weight:
            for idx in range(len(consis_vio)):
                vio_val_curr: float = consis_vio[idx]
                vio_val_prev: float = consis_vio_prev[idx]
                if abs(vio_val_curr) > self._weight_update_fraction * abs(vio_val_prev):
                    penalty_weight[idx] *= self._weight_update_coefficient

        updated_params: dict = {
            "lagrange est": lagrange_est,
            "penalty weight": penalty_weight,
        }
        return updated_params
