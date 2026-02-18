"""
Solve subproblems in ALC inner loops
"""

from __future__ import annotations
from typing import Any
from .type_defs import SubproblemResult, SubprobDict, SolveSubproblemFunc
import numpy as np


def solve_subproblem(
    target_shared_var: np.ndarray,
    lagrange_est: np.ndarray,
    penalty_weight: np.ndarray,
    subprob_dict: SubprobDict,
    local_var_idx: list[int],
    aux_shared_var_idx: list[int],
    initial_guess: np.ndarray | None = None,
    args: Any | None = None,
) -> SubproblemResult:
    """initial guess, shared var, etc must be in the compatible dim for subproblems

    Args:
        target_shared_var: Shared variable vector calculatd by master problem
        lagrange_est: Lagrange multiplier estimate
        penalty_weight: Penalty weight for discprepancy
        subprob_dict: Subproblem dictionary containing
            opimization problem type (MIP or NLP)
            solver function returning objective function value and design variable
            solver arguments as list
        initial_guess: design variable initial guess for original AIO problem
    Returns:
        objective: Objective function value
        design var: Design variable vector
    """

    subp_solve_func: SolveSubproblemFunc = subprob_dict["function"]
    optim_type: str = subprob_dict["optim type"]

    if optim_type not in ["MIP", "NLP"]:
        raise ValueError("""Invalid optimization problem type.
        It must be either "MIP" or "NLP".""")
    if optim_type == "NLP":
        if initial_guess is None:
            raise ValueError("Initial guess must be provided for NLP subproblems.")
        subp_initial_guess = initial_guess[local_var_idx]
    else:
        subp_initial_guess = None

    subp_result: SubproblemResult = subp_solve_func(
        target_shared_var=target_shared_var,
        lagrange_est=lagrange_est,
        penalty_weight=penalty_weight,
        local_var_idx=local_var_idx,
        aux_shared_var_idx=aux_shared_var_idx,
        initial_guess=subp_initial_guess,
        args=subprob_dict.get("args"),
    )

    return subp_result
