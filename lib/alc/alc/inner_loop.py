import numpy as np
# import multiprocessing as mP

from .dimension_converter import DimensionConverter
from .type_defs import (
    SolveSubproblemFuncWrapper,
    SubproblemResult,
    AllSubpResults,
    AllSubpDict,
)
import copy
from icecream import ic


class InnerLoop:
    def __init__(
        self,
        dc: DimensionConverter,
        all_subprob_dict: AllSubpDict,
        initial_guess: np.ndarray,
        use_admm: bool = False,
        tol_inner: float = 1e-5,
        prioritized_var: tuple = (),
        parallel_mode: bool = False,
        verbose: bool = False,
        solve_fn: SolveSubproblemFuncWrapper | None = None,
    ) -> None:
        """Initialize ALC inner loop

        Args:
            dc: DimensionConverter class instance
            all_subprob_dict: Dictionary containing all subproblem information
            initial_guess: Initial guess for the shared variables
            use_admm: ADMM flag, skips inner loop if True. Default False
            tol_inner: Tolerance for the inner loop
            parallel_mode: Parallel mode flag, default False

            lagrange_est_list: List of Lagrange estimates for each subproblem
            penalty_weight_list: List of penalty weights for each subproblem
            outer_counter: Outer loop counter
        """

        self._dc = dc
        self._all_subprob_dict = all_subprob_dict
        self._initial_guess = initial_guess
        self._tol_inner = tol_inner
        self._prioritized_var = prioritized_var
        self._parallel_mode = parallel_mode
        self._n_subprob = self._dc.n_subprob
        self._use_admm = use_admm
        self._verbose = verbose
        if solve_fn is None:
            from .subproblems import solve_subproblem as solve_func

            self._solve_func: SolveSubproblemFuncWrapper = solve_func
        else:
            self._solve_func: SolveSubproblemFuncWrapper = solve_fn
        # defined later with setter
        self._lagrange_est_list: list[np.ndarray] | None = None
        self._penalty_weight_list: list[np.ndarray] | None = None
        self._outer_counter: int | None = None

    @property
    def lagrange_est_list(self):
        return self._lagrange_est_list

    @lagrange_est_list.setter
    def lagrange_est_list(self, lagrange_est_list: list[np.ndarray]):
        self._check_arg_is_list_of_ndarrays(lagrange_est_list)
        self._lagrange_est_list = lagrange_est_list

    @property
    def penalty_weight_list(self):
        return self._penalty_weight_list

    @penalty_weight_list.setter
    def penalty_weight_list(self, penalty_weight_list: list[np.ndarray]):
        self._check_arg_is_list_of_ndarrays(penalty_weight_list)
        self._penalty_weight_list = penalty_weight_list

    @property
    def outer_counter(self):
        return self._outer_counter

    @outer_counter.setter
    def outer_counter(self, outer_counter: int):
        assert isinstance(outer_counter, int)
        self._outer_counter = outer_counter

    def run(self) -> dict[str, np.ndarray]:
        """Run ALC inner loop

        1. solve all subproblems with the current shared variable vector
        2. check for inner loop convergence (may be skipped)
        3. Calculates the new target shared variable with the master problem
        4. Go back to step 1 if not converged (may be skipped)

        Returns:
            Dictionary containing the inner loop results
                "shared": shared variable vector
                "aux shared": auxiliary shared variable vector
                "all": all variable vector
                "objectives": list of objective values for each subproblem
        """

        self._ensure_setters_used()

        inner_counter: int = 0
        relaxed_obj_sum_improvement: float = 0
        # variable innter loop tolerance
        tol_inner_current = max(self._tol_inner, 10 ** (-0.5 * self._outer_counter))
        shared_var = self._dc.all_to_global_shared(self._initial_guess)

        if self._use_admm:
            sp_results: AllSubpResults = self._solve_all_subproblems(shared_var)
            local_obj = sp_results["local obj"]
            aux_shared_var = sp_results["aux shared var"]
            shared_var = self._solve_master_problem(aux_shared_var)
        else:
            while inner_counter <= 1 or relaxed_obj_sum_improvement > tol_inner_current:
                inner_counter += 1
                if self._verbose:
                    print("### Inner Loop Iteration: ", inner_counter)

                sp_results = self._solve_all_subproblems(shared_var)
                local_obj = sp_results["local obj"]
                aux_shared_var = sp_results["aux shared var"]

                # Check improvement from previsous inter loop iteration
                if inner_counter == 1:
                    current_relaxed_obj_sum: float = sum(local_obj)
                if inner_counter >= 2:
                    prev_relaxed_obj_sum: float = current_relaxed_obj_sum
                    current_relaxed_obj_sum: float = sum(local_obj)
                    relaxed_obj_sum_improvement: float = abs(
                        current_relaxed_obj_sum - prev_relaxed_obj_sum
                    ) / (1 + abs(current_relaxed_obj_sum))
                    if self._verbose:
                        ic(relaxed_obj_sum_improvement)

                shared_var = self._solve_master_problem(aux_shared_var)
                if self._verbose:
                    ic(shared_var)

        self._initial_guess = self._update_initial_guess(
            shared_var, sp_results["local design var"]
        )

        inner_loop_results: dict[str, np.ndarray | list[np.ndarray] | list[float]] = {
            "shared": shared_var,
            "aux shared": aux_shared_var,
            "all": copy.deepcopy(self._initial_guess),
            "objectives": local_obj,
        }

        return inner_loop_results

    def _solve_all_subproblems(
        self,
        shared_var: np.ndarray,
    ) -> AllSubpResults:
        """solve all subproblems

        Args:
            shared_var: shared variable vector
        Returns:
            all_sp_results: dictionary containing the results of the subproblems
                "local design var": list of local design variable vectors
                "local obj": list of local objective function values
                "aux shared var": list of auxiliary shared variable vectors
        """

        local_design_var: list[np.ndarray] = [
            np.zeros(dim) for dim in self._dc.dim_local_var_list
        ]
        local_obj: list[float] = [0] * self._n_subprob
        aux_shared_var: list[np.ndarray] = [
            np.zeros(dim) for dim in self._dc.dim_aux_shared_var
        ]

        if self._parallel_mode:
            raise NotImplementedError("Parallel mode not implemented yet")
            # TODO: implement parallel mode

            # pool1 = mP.Pool(self._n_subprob)
            # inputs = [(sp_id, self._initial_guess, shared_var, v[sp_id], w[sp_id], x2xj[sp_id], y2yj[sp_id],
            #            xj2x[sp_id], x2yj[sp_id], xjidx[sp_id]) for sp_id in range(self._n_subprob)]
            # result = pool1.map(solve_subproblems_wrapper, inputs)
            # pool1.close()

        for sp_id in range(self._n_subprob):
            sp_result: SubproblemResult = self._solve_func(
                target_shared_var=self._dc.shared_to_aux_shared(shared_var, sp_id),
                lagrange_est=self._lagrange_est_list[sp_id],
                penalty_weight=self._penalty_weight_list[sp_id],
                subprob_dict=self._all_subprob_dict[sp_id],
                local_var_idx=self._dc.local_var_idx_list[sp_id],
                aux_shared_var_idx=self._dc.aux_shared_var_idx_list[sp_id],
                initial_guess=self._initial_guess,
            )
            local_obj[sp_id] = sp_result["objective"]
            local_design_var[sp_id] = sp_result["design var"]
            aux_shared_var[sp_id] = self._dc.local_to_aux_shared(
                local_design_var[sp_id], sp_id
            )
            if self._verbose:
                ic(sp_id)
                ic(local_obj[sp_id])
                ic(local_design_var[sp_id])

        all_sp_results: AllSubpResults = {
            "local design var": local_design_var,
            "local obj": local_obj,
            "aux shared var": aux_shared_var,
        }
        return all_sp_results

    def _solve_master_problem(self, aux_shared_var: list[np.ndarray]) -> np.ndarray:
        """Master Problem: Calculate new shared variables based on the subproblem solutions

        Analytical expression for the new shared variable is decomposed into three terms,
        which also need to be calculated separately for each subproblem

        Args:
            aux_shared_var: Auxiliary shared variable vector for all subproblems
        Retruns:
            shared_var: Newly calculated shared variable vector
        """

        term1, term2, term3 = 0, 0, 0
        for subprob_idx in range(self._n_subprob):
            lagrange_est: np.ndarray = np.array(self._lagrange_est_list[subprob_idx])
            penalty_weight: np.ndarray = np.array(
                self._penalty_weight_list[subprob_idx]
            )
            term1 += self._dc.aux_shared_to_shared(
                (penalty_weight * penalty_weight * aux_shared_var[subprob_idx]),
                subprob_idx,
            )
            term2 += self._dc.aux_shared_to_shared(lagrange_est, subprob_idx)
            term3 += self._dc.aux_shared_to_shared(
                (penalty_weight * penalty_weight), subprob_idx
            )
        shared_var: np.ndarray = (term1 - 0.5 * term2) / term3
        shared_var: np.ndarray = shared_var.flatten()  # (ny,1) -> (ny,)

        if self._prioritized_var:
            for pri_var_dict in self._prioritized_var:
                pri_var_idx: int = pri_var_dict["prioritized var idx"]
                pri_supb_id: int = pri_var_dict["prioritized subp id"]
                idx_in_subp: int = self._dc.aux_shared_var_idx_list[pri_supb_id].index(
                    pri_var_idx
                )
                shared_var[pri_var_idx] = aux_shared_var[pri_supb_id][idx_in_subp]
        return shared_var

    def _update_initial_guess(
        self,
        shared_var: np.ndarray,
        local_design_var: list[np.ndarray],
    ) -> np.ndarray:
        """update the initial guess for the next outer loop's first inner loop

        It first expands up all local variable vector dimension to
        all variable vector dimension, and then sum them up.
        If a variable is stricly local to a subproblem, the summed up vector contains the correct value. If a variable is shared, it is updated with the shared variable vector
        which are calculated by the master problem.

        Args:
            shared_var: shared variable vector
            local_design_var: list of local design variable vectors
        Retruns:
            initial_guess: updated initial guess
        """
        initial_guess: np.ndarray = np.zeros(self._dc.dim_all_var)
        for sp_id in range(self._n_subprob):
            sparse_local_var: np.ndarray = self._dc.local_to_all(
                local_design_var[sp_id], sp_id
            )
            initial_guess += sparse_local_var
        for idx in range(self._dc.dim_shared_var):
            corresp_idx: int = self._dc.shared_var_idx[idx]
            initial_guess[corresp_idx] = shared_var[idx]
        return initial_guess

    def _check_arg_is_list_of_ndarrays(self, arg):
        """Check if argument is a list of 1D numpy arrays
        Elements of the list should be in shape (n,) not (n,1)
        """
        assert isinstance(arg, list)
        for elem in arg:
            assert isinstance(elem, np.ndarray)
            assert len(elem.shape) == 1

    def _ensure_setters_used(self) -> None:
        """Ensure setters are used, and inst. var. are not None"""
        if any(
            [
                self._lagrange_est_list is None,
                self._penalty_weight_list is None,
                self._outer_counter is None,
            ]
        ):
            raise RuntimeError(
                "Lagrange multiplier estimate, penalty weight,"
                "and outer loop counter must be set via setter."
                "(They are not instanced in the initialization.)"
            )
