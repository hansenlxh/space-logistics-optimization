"""
Solve subproblems in ALC inner loops
"""

import numpy as np
import pygmo as pg

# import pygmo_plugins_nonfree as ppnf  # only needed if snopt is used
import os
import sys
from typing import Callable
from ..type_defs import SubproblemResult, SubprobDict, SolveSubproblemFunc
from icecream import ic

sys.path.append("../libsnopt7")  # make snopt license file discoverable


def solve_subp_pygmo_wrapper(
    target_shared_var: np.ndarray,
    lagrange_est: np.ndarray,
    penalty_weight: np.ndarray,
    subprob_dict: SubprobDict,
    local_var_idx: list[int],
    aux_shared_var_idx: list[int],
    initial_guess: np.ndarray | None = None,
    args=None,
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
        use_mbh: Monotoic Basin Hopping flag for NLP subproblems, default False
    Returns:
        obj: Objective function value
        design_var: Design variable vector
    """
    use_mbh = False
    solver_func: Callable = subprob_dict["function"]
    optim_type: str = subprob_dict["optim type"]

    if optim_type not in ["MIP", "NLP"]:
        raise ValueError("""Invalid optimization problem type.
        It must be either "MIP" or "NLP".""")
    if optim_type == "NLP":
        if initial_guess is None:
            raise ValueError("Initial guess must be provided for NLP subproblems.")
        subp_res: SubproblemResult = solve_nlp_subp_w_pygmo(
            target_shared_var,
            lagrange_est,
            penalty_weight,
            local_var_idx,
            aux_shared_var_idx,
            initial_guess=initial_guess[local_var_idx],
            args=subprob_dict.get("args"),
        )
        return subp_res
    else:
        subp_res: SubproblemResult = solver_func(
            target_shared_var,
            lagrange_est,
            penalty_weight,
            local_var_idx,
            aux_shared_var_idx,
            subprob_dict.get("args"),
        )
        return subp_res


def solve_nlp_subp_w_pygmo(
    target_shared_var: np.ndarray,
    lagrange_est: np.ndarray,
    penalty_weight: np.ndarray,
    local_var_idx: list[int],
    aux_shared_var_idx: list[int],
    initial_guess: np.ndarray | None = None,
    args=None,
) -> SubproblemResult:
    """Solves NLP subproblem using pygmo
    Args:
        pygmo_udp: pygmo user defined problem (class)
        initial_guess: initial guess for subrproblem's local variables
    Returns:
        obj: Objective function value
        design_var: Design variable vector
    """
    pygmo_udp = args
    if pygmo_udp is None:
        raise ValueError("""
        To use pygmo, pygmo user-defined problem must be passed as
        an optional argument when constructing subproblem dictionary.
        """)
    pygmo_udp = pygmo_udp(
        target_shared_var,
        lagrange_est,
        penalty_weight,
        local_var_idx,
        aux_shared_var_idx,
        initial_guess,
    )
    use_mbh = False

    snopt_lib = os.path.dirname(os.getcwd())
    snopt_lib = os.path.join(os.path.join(snopt_lib, "libsnopt7"), "snopt7.dll")
    algo = _get_algorithm(
        solver="ipopt",
        maxiter=1000,
        max_cpu_time=1000,
        snopt_lib=snopt_lib,
        use_mbh=use_mbh,
    )
    prob = pg.problem(pygmo_udp)
    prob.c_tol = 1e-8
    initial_pop = pg.population(prob, size=0)
    initial_pop.push_back(x=_offset_initial_guess(pygmo_udp, initial_guess))
    evolved_pop = algo.evolve(initial_pop)
    best_idx = evolved_pop.best_idx()
    # [0] needed here because pygmo supports multi-objective optimization
    # and returns a list of objectives
    obj = evolved_pop.get_f()[best_idx][0]
    local_design_var = evolved_pop.get_x()[best_idx]

    res = {"objective": obj, "design var": local_design_var}
    return res


def _offset_initial_guess(pygmo_udp, initial_guess: np.ndarray) -> np.ndarray:
    """Prevent out-of-bound initial guess

    Modifies (if necessary) and returns initial guess to prevent
    pygmo error occurred when an individual whose chromosome is
    out of bound is added to population.
    It seems pygmo checks the bounds very strictly, so the initial guess
    updated by numerical solver solutions can violate bounds.
    (e.g, 1.99999 returns error if its lower bound is 2)

    Args:
        pygmo_udp: pygmo user defined problem (class)
        initial_guess: initial guess for subrproblem's local variables
    Returns:
        initial_guess: offset initial guess
    """
    lb: list[float] = pygmo_udp.get_bounds()[0]
    ub: list[float] = pygmo_udp.get_bounds()[1]
    while any(initial_guess - ub > 0):
        for idx in np.where(initial_guess - ub > 0):
            initial_guess[idx] -= 1e-3
    while any(lb - initial_guess > 0):
        for idx in np.where(lb - initial_guess > 0):
            initial_guess[idx] += 1e-3
    return initial_guess


def _get_algorithm(
    solver: str = "snopt",
    use_mbh: bool = False,
    ftol: float = 1e-8,
    ctol: float = 1e-8,
    maxiter: int = 1000,
    max_cpu_time: float = 1000,
    snopt_lib=None,
) -> pg.algorithm:
    """Function to set up algorithm in pygmo

    Args:
        solver:     solver to be used
        use_mbh:    use multi-start hill climbing
        ftol:       relative convergence tolerance
        ctol:       constraint tolerance
        maxiter:    maximum number of iterations
        max_cpu_time: maximum cpu time
        snosnopt_lib: path to snopt7.dll file
    Returns:
        algo:       pygmo algorithm
    """
    if solver == "ipopt":
        ip = pg.ipopt()
        # Change the relative convergence tolerance
        ip.set_numeric_option("tol", ftol)
        ip.set_numeric_option("constr_viol_tol", ctol)
        ip.set_numeric_option("max_cpu_time", max_cpu_time)
        if use_mbh:
            # wrap uda with mbh
            uda = pg.algorithm(ip)
            algo = pg.algorithm(pg.mbh(uda, stop=5, perturb=0.1))
        else:
            algo = pg.algorithm(ip)
            algo.set_verbosity(0)

    elif solver == "snopt":
        if snopt_lib is None:
            raise ValueError("Provide path to snopt7.dll file!")
        # use snopt
        pygmoSnopt = ppnf.snopt7(
            screen_output=False, library=snopt_lib, minor_version=7
        )
        pygmoSnopt.set_numeric_option("Major optimality tolerance", ftol)
        pygmoSnopt.set_numeric_option("Major feasibility tolerance", ctol)
        pygmoSnopt.set_numeric_option("Minor feasibility tolerance", ctol)
        pygmoSnopt.set_numeric_option("Major step limit", 2)
        pygmoSnopt.set_integer_option("Iterations limit", maxiter)  # should be 4e3
        if use_mbh:
            # wrap uda with mbh
            uda = pg.algorithm(pygmoSnopt)
            algo = pg.algorithm(pg.mbh(uda, stop=5, perturb=0.1))
            algo.set_verbosity(0)
        else:
            # pure snopt
            algo = pg.algorithm(pygmoSnopt)
            algo.set_verbosity(0)
    else:
        raise ValueError("Invalid solver choice. It must be either 'ipopt' or 'snopt'.")

    return algo
