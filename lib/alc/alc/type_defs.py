from __future__ import annotations
from typing import Protocol, TypedDict, Literal, NotRequired, Any, TypeAlias
import numpy as np

SubproblemResult = TypedDict(
    "SubproblemResult",
    {
        "objective": float,
        "design var": np.ndarray,
    },
)

AllSubpResults = TypedDict(
    "AllSubpResults",
    {
        "local design var": list[np.ndarray],
        "local obj": list[float],
        "aux shared var": list[np.ndarray],
    },
)


class SolveSubproblemFuncWrapper(Protocol):
    def __call__(
        self,
        target_shared_var: np.ndarray,
        lagrange_est: np.ndarray,
        penalty_weight: np.ndarray,
        subprob_dict: SubprobDict,
        local_var_idx: list[int],
        aux_shared_var_idx: list[int],
        initial_guess: np.ndarray | None = None,
        args: Any | None = None,
    ) -> SubproblemResult: ...


class SolveSubproblemFunc(Protocol):
    def __call__(
        self,
        target_shared_var: np.ndarray,
        lagrange_est: np.ndarray,
        penalty_weight: np.ndarray,
        local_var_idx: list[int],
        aux_shared_var_idx: list[int],
        initial_guess: np.ndarray | None = None,
        args: Any | None = None,
    ) -> SubproblemResult: ...


SubprobDict = TypedDict(
    "SubprobDict",
    {
        "optim type": Literal["MIP", "NLP"],
        "function": SolveSubproblemFunc,
        "args": NotRequired[Any],
    },
)

AllSubpDict: TypeAlias = dict[int, "SubprobDict"]
