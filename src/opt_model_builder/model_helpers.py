import numpy as np


def get_dependency_matrix(self) -> list[list[int]]:
    """returns the dependency matrix for the subproblem

    Returns:
        dependency_matrix: dependency matrix where
            1 represents the shared variable is used in the subproblem
            0 otherwise
    """
    dep_matrix: list = []
    n_subprob: int = 1 + self.n_sc_design  # + self.n_isru_design
    n_shared_vars: int = (
        self.n_sc_design * self.n_sc_vars  # + self.n_isru_design * self.n_isru_vars
    )
    sc_subprob_idx = list(range(1, self.n_sc_design + 1))
    isru_subprob_idx = list(range(self.n_sc_design + 1, n_subprob))
    for sp_id in range(n_subprob):
        dep_vec = np.zeros(n_shared_vars, dtype=int)
        if sp_id == 0:  # space logistics subproblem
            dep_vec += 1

        if sp_id in sc_subprob_idx:
            sc_des = sp_id  # sc_des ∈ {1,2,..., self.n_sc_design}
            start_index = (sc_des - 1) * self.n_sc_vars
            end_index = start_index + self.n_sc_vars
            dep_vec[start_index:end_index] = 1

        # if sp_id in isru_subprob_idx:
        #     # isru_id ∈ {1,2,..., self.n_isru_design}
        #     isru_id = sp_id - self.n_sc_design
        #     start_index = (
        #         self.n_sc_design * self.n_sc_vars +
        #         (isru_id - 1) * self.n_isru_vars
        #     )
        #     end_index = start_index + self.n_isru_vars
        #     dep_vec[start_index:end_index] = 1

        dep_matrix.append(dep_vec)
    dep_matrix = np.asarray(dep_matrix).tolist()
    return dep_matrix
