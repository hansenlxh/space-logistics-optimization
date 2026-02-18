import numpy as np


class DimensionConverter:
    def __init__(
        self,
        dependency_matrix: list[list[int]],
        dim_all_var: int,
    ) -> None:
        """Initialize dimension converter
        Args:
            dependency_matrix (list[list[int]]):    Dependency matrix; each row contains
                a subproblem's dependency on design variables.
                1: dependent, 0: independent
            dim_all_var (int):  Dimension of all variables
        """
        self._dependency_matrix = np.array(dependency_matrix)
        self.dim_all_var = dim_all_var
        self.n_subprob = len(dependency_matrix)
        # placeholder
        self.dim_local_var_list: list[int] = [0] * self.n_subprob
        self.dim_shared_var: int = 0
        self.dim_aux_shared_var: list[int] = [0] * self.n_subprob
        self.local_var_idx_list: list[list[int]] = []
        self.shared_var_idx: list[int] = []
        self.aux_shared_var_idx_list: list[list[int]] = []
        # emulating __post_init__ from dataclasses
        self._post_init()

    def _post_init(self) -> None:
        """Sanity check and computation of private instance variables"""

        assert self.dim_all_var > 0, """
        The dimenstion of design variables must be positive. Received value: {}""".format(
            self.dim_all_var
        )

        assert all(len(row) == self.dim_all_var for row in self._dependency_matrix), """
        Each row of dependency matrix (corresponding to each subproblem)
        must have the same dimension as the number of design variables.
        Received dependency matrix: {}\n Dimension of variables: {}
        """.format(self._dependency_matrix, self.dim_all_var)

        # compute private instance variables just once at class initialization
        self._compute_conversion_matrices()

    def all_to_global_shared(self, all_var: np.ndarray) -> np.ndarray:
        """Convert dimension of all variables to global shared variables
        Args:
            all_var (np.ndarray): all design variables,
                including shared and local variables for all subproblems
        Returns:
            shared_var (np.ndarray): global shared design variables
        """
        return self._x2y @ all_var

    def all_to_aux_shared(self, all_var: np.ndarray, subprob_idx: int) -> np.ndarray:
        """Convert dimension of all variables to auxiliary shared variables
        Args:
            all_var (np.ndarray): all design variables,
                including shared and local variables for all subproblems
            subprob_idx (int): index of subproblem
        Returns:
            aux_shared_var (np.ndarray): auxiliary shared design variables
        """
        return self._x2yj[subprob_idx] @ all_var

    def all_to_local(self, all_var: np.ndarray, subprob_idx: int) -> np.ndarray:
        """Convert dimension of all variables to local variables for a subproblem
        Args:
            all_var (np.ndarray): all design variables,
                including shared and local variables for all subproblems
            subprob_idx (int): index of subproblem
        Returns:
            local_var (np.ndarray): local design variables (incl. shared variables) for a subproblem
        """
        return self._x2xj[subprob_idx] @ all_var

    def local_to_all(self, local_var: np.ndarray, subprob_idx: int) -> np.ndarray:
        """Convert dimension of local variables for a subproblem to all variables
        Args:
            local_var (np.ndarray): local design variables (incl. shared variables) for a subproblem
            subprob_idx (int): index of subproblem
        Returns:
            all_var (np.ndarray): all design variables,
                including shared and local variables for all subproblems
        """
        return self._xj2x[subprob_idx] @ local_var

    def shared_to_aux_shared(
        self, shared_var: np.ndarray, subprob_idx: int
    ) -> np.ndarray:
        """Convert dimension of shared variables to auxiliary shared variables
        Args:
            shared_var (np.ndarray): global shared design variables
            subprob_idx (int): index of subproblem
        Returns:
            aux_shared_var (np.ndarray): auxiliary shared design variables
        """
        return self._y2yj[subprob_idx] @ shared_var

    def aux_shared_to_shared(
        self, aux_shared_var: np.ndarray, subprob_idx: int
    ) -> np.ndarray:
        """Convert dimension of auxiliary shared variables to shared variables
        Args:
            aux_shared_var (np.ndarray): auxiliary shared design variables
            subprob_idx (int): index of subproblem
        Returns:
            shared_var (np.ndarray): global shared design variables
        """
        return self._yj2y[subprob_idx] @ aux_shared_var

    def local_to_aux_shared(
        self, local_var: np.ndarray, subprob_idx: int
    ) -> np.ndarray:
        """Convert dimension of local variables to auxiliary shared variables
        Args:
            local_var (np.ndarray): local design variables (incl. shared variables) for a subproblem
            subprob_idx (int): index of subproblem
        Returns:
            aux_shared_var (np.ndarray): auxiliary shared design variables
        """
        return self._x2yj[subprob_idx] @ (self._xj2x[subprob_idx] @ local_var)

    def _compute_conversion_matrices(self) -> None:
        """Compute conversion matrices"""
        # TODO: change the variable names (they're not intuitive and confusing)
        # Also missing y2x? (Do I even need it?)

        temp_design_var = np.ones(self.dim_all_var)
        """x2y"""
        Ssum = np.zeros(self.dim_all_var)
        for j in range(self.n_subprob):
            Ssum = Ssum + self._dependency_matrix[j]
        self.shared_var_idx = np.argwhere(Ssum >= 2).flatten().tolist()
        self.dim_shared_var = len(self.shared_var_idx)

        self._x2y = np.zeros((self.dim_shared_var, self.dim_all_var))
        for i in range(self.dim_shared_var):
            self._x2y[i, :] = 0
            self._x2y[i, self.shared_var_idx[i]] = 1

        """x2yj"""
        self._x2yj = []
        for j in range(self.n_subprob):
            jj = [*range(self.n_subprob)]
            jj.pop(j)
            idx = np.array([])
            for j2 in jj:
                Stemp = self._dependency_matrix[j] + self._dependency_matrix[j2]
                idx2 = np.argwhere(Stemp == 2)
                if len(idx) == 0:
                    idx = idx2
                else:
                    idx = np.concatenate((idx, idx2))
            idx = np.transpose(idx)
            idx = np.unique(idx).tolist()
            self.aux_shared_var_idx_list.append(idx)

            Syj = np.zeros((len(idx), self.dim_all_var))
            # print(Syj.shape)
            for i in range(len(idx)):
                Syj[i, :] = 0
                Syj[i, idx[i]] = 1
            Syj = Syj[np.any(Syj, axis=1), :]
            self._x2yj.append(Syj)
            self.dim_aux_shared_var[j] = len(self._x2yj[j] @ temp_design_var)

        """xj2x"""
        xj = []
        xjidx2 = []
        self._xj2x = []
        self.dim_local_var_list: list[int] = [0] * self.n_subprob
        for j in range(self.n_subprob):
            xjtemp = temp_design_var * self._dependency_matrix[j]
            a = list(np.ravel(np.argwhere(xjtemp == 0)))
            xjtemp = np.delete(xjtemp, a)
            if len(xj) == 0:
                xj = xjtemp
            else:
                xj = np.concatenate((xj, xjtemp))
            self.dim_local_var_list[j] = len(xjtemp)
            if len(self.local_var_idx_list) == 0:
                self.local_var_idx_list = list(
                    [np.ravel(np.argwhere(self._dependency_matrix[j] == 1))]
                )
            else:
                xjidx_temp = list(
                    [np.ravel(np.argwhere(self._dependency_matrix[j] == 1))]
                )
                self.local_var_idx_list = self.local_var_idx_list + xjidx_temp

        for j in range(self.n_subprob):
            k = 1
            Temp = [
                [0 for x in range(self.dim_local_var_list[j])]
                for y in range(self.dim_all_var)
            ]
            idx = self.local_var_idx_list[j]
            idx = list(map(int, idx))
            for i in range(self.dim_all_var):
                if i == idx[i]:
                    Temp[i][k - 1] = 1
                    k += 1
                    if k > self.dim_local_var_list[j]:
                        if i + 1 < self.dim_all_var:
                            idx = np.concatenate(
                                (idx, np.zeros(self.dim_all_var - (i + 1)))
                            )
                            break
                        else:
                            break
                elif i == 0:
                    idx.insert(0, 0)
                else:
                    idx.insert(i, 0)

            # idx = list(np.transpose(idx))
            if len(xjidx2) == 0:
                xjidx2 = [list(map(int, idx))]
            else:
                xjidx2.append(list(map(int, idx)))

            if self._xj2x == []:
                self._xj2x = [Temp]
            else:
                self._xj2x.append(Temp)

        """yj2y"""
        yjidx2 = []
        self._yj2y = []
        for j in range(self.n_subprob):
            k = 1
            Temp = [
                [0 for x in range(self.dim_aux_shared_var[j])]
                for y in range(self.dim_shared_var)
            ]
            idx = self.aux_shared_var_idx_list[j]
            idx = list(map(int, idx))
            for i in range(self.dim_shared_var):
                if self.shared_var_idx[i] == idx[i]:
                    Temp[i][k - 1] = 1
                    k += 1
                    if k > self.dim_aux_shared_var[j]:
                        if i + 1 < self.dim_shared_var:
                            idx = np.concatenate(
                                (idx, np.zeros(self.dim_shared_var - (i + 1)))
                            )
                            break
                        else:
                            break
                elif i == 0:
                    idx.insert(0, 0)
                else:
                    idx.insert(i, 0)

            # idx = list(np.transpose(idx))
            if len(yjidx2) == 0:
                yjidx2 = [list(map(int, idx))]
            else:
                yjidx2.append(list(map(int, idx)))

            Temp = np.array(Temp)

            if self._yj2y == []:
                self._yj2y = [Temp]
            else:
                self._yj2y.append(Temp)

        """x2xj and y2yj"""
        self._x2xj = []
        self._y2yj = []
        for j in range(self.n_subprob):
            xj2x_temp = np.array(self._xj2x[j])
            yj2y_temp = np.array(self._yj2y[j])
            if self._x2xj == []:
                self._x2xj = [np.transpose(xj2x_temp)]
                self._y2yj = [np.transpose(yj2y_temp)]
            else:
                self._x2xj.append(np.transpose(xj2x_temp))
                self._y2yj.append(np.transpose(yj2y_temp))

        for j in range(self.n_subprob):
            self.local_var_idx_list[j] = self.local_var_idx_list[j].tolist()
