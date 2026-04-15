"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest

from inflammation.models import daily_mean, daily_max, daily_min

@pytest.mark.parametrize(
    "values, expected",
    [
        ([ [0, 0], [0, 0], [0, 0] ], [0, 0]),
        ([ [1, 2], [3, 4], [5, 6] ], [3, 4]),
    ])
def test_daily_mean_zeros(values, expected):
    """Test that mean function works."""

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(np.array(values)), np.array(expected))


@pytest.mark.parametrize(
    "values, expected",
    [
        ([ [1, 2], [3, 7], [5, 4] ], [5, 7]),
        ([ [0, 0], [0, 0], [0, 0] ], [0, 0]),
    ]
)
def test_daily_max(values, expected):
    """Test that max function works."""

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_max(values), expected)


@pytest.mark.parametrize(
    "values, expected",
    [
        ([ [6, 2], [3, 7], [5, 4] ], [3, 2]),
        ([ [0, 0], [0, 0], [0, 0] ], [0, 0]),
    ]
)
def test_daily_min(values, expected):
    """Test that min function works."""

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_min(values), expected)
