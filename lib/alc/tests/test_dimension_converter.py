try:
    # if used as a subtree
    from lib.alc.alc import DimensionConverter
except (ModuleNotFoundError, ImportError):
    # for stant-alone testing
    from alc import DimensionConverter
import numpy as np
import pytest

dep_matrix = [
    [1, 1, 1, 0, 0],
    [0, 0, 1, 1, 0],
    [0, 1, 0, 1, 1],
]
dim_all_vars = 5
all_var = np.array([1, 2, 3, 4, 5])
shared_var = np.array([2, 3, 4])


@pytest.fixture(scope="module")
def dc_inst():
    return DimensionConverter(dep_matrix, dim_all_vars)


def test_all_to_global_shared(dc_inst):
    result = dc_inst.all_to_global_shared(all_var)
    np.testing.assert_array_almost_equal(
        np.array([2, 3, 4]),
        result,
        decimal=6,
    )


@pytest.mark.parametrize(
    "subprob_idx, expected",
    [
        (0, np.array([2, 3])),
        (1, np.array([3, 4])),
        (2, np.array([2, 4])),
    ],
)
def test_all_to_aux_shared(dc_inst, subprob_idx, expected):
    result = dc_inst.all_to_aux_shared(all_var, subprob_idx)
    np.testing.assert_array_almost_equal(
        expected,
        result,
        decimal=6,
    )


@pytest.mark.parametrize(
    "subprob_idx, expected",
    [
        (0, np.array([1, 2, 3])),
        (1, np.array([3, 4])),
        (2, np.array([2, 4, 5])),
    ],
)
def test_all_to_local(dc_inst, subprob_idx, expected):
    result = dc_inst.all_to_local(all_var, subprob_idx)
    np.testing.assert_array_almost_equal(
        expected,
        result,
        decimal=6,
    )


@pytest.mark.parametrize(
    "subprob_idx, local_var, expected",
    [
        (0, np.array([1, 2, 3]), np.array([1, 2, 3, 0, 0])),
        (1, np.array([3, 4]), np.array([0, 0, 3, 4, 0])),
        (2, np.array([2, 4, 5]), np.array([0, 2, 0, 4, 5])),
    ],
)
def test_local_to_all(dc_inst, subprob_idx, local_var, expected):
    result = dc_inst.local_to_all(local_var, subprob_idx)
    np.testing.assert_array_almost_equal(
        expected,
        result,
        decimal=6,
    )


@pytest.mark.parametrize(
    "subprob_idx, expected",
    [
        (0, np.array([2, 3])),
        (1, np.array([3, 4])),
        (2, np.array([2, 4])),
    ],
)
def test_shared_to_aux_shared(dc_inst, subprob_idx, expected):
    result = dc_inst.shared_to_aux_shared(shared_var, subprob_idx)
    np.testing.assert_array_almost_equal(
        expected,
        result,
        decimal=6,
    )


@pytest.mark.parametrize(
    "subprob_idx, aux_shared_var, expected",
    [
        (0, np.array([2, 3]), np.array([2, 3, 0])),
        (1, np.array([3, 4]), np.array([0, 3, 4])),
        (2, np.array([2, 4]), np.array([2, 0, 4])),
    ],
)
def test_aux_shared_to_shared(dc_inst, subprob_idx, aux_shared_var, expected):
    result = dc_inst.aux_shared_to_shared(aux_shared_var, subprob_idx)
    np.testing.assert_array_almost_equal(
        expected,
        result,
        decimal=6,
    )


@pytest.mark.parametrize(
    "subprob_idx, local_var, expected",
    [
        (0, np.array([1, 2, 3]), np.array([2, 3])),
        (1, np.array([3, 4]), np.array([3, 4])),
        (2, np.array([2, 4, 5]), np.array([2, 4])),
    ],
)
def test_local_to_aux_shared(dc_inst, subprob_idx, local_var, expected):
    result = dc_inst.local_to_aux_shared(local_var, subprob_idx)
    np.testing.assert_array_almost_equal(
        expected,
        result,
        decimal=6,
    )


def test_dimensions(dc_inst):
    assert dc_inst.dim_local_var_list == [3, 2, 3]
    assert dc_inst.dim_shared_var == 3
    assert dc_inst.dim_aux_shared_var == [2, 2, 2]


def test_indicies(dc_inst):
    assert dc_inst.local_var_idx_list == [[0, 1, 2], [2, 3], [1, 3, 4]]
    assert dc_inst.shared_var_idx == [1, 2, 3]
    assert dc_inst.aux_shared_var_idx_list == [[1, 2], [2, 3], [1, 3]]


def test_n_subprob(dc_inst):
    assert dc_inst.n_subprob == 3
