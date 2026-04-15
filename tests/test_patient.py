"""Tests for the Patient model."""

import numpy as np
import numpy.testing as npt
import pytest

from contextlib import nullcontext

from inflammation.models import Patient, patient_normalise

def test_create_patient():

    name = 'Alice'
    p = Patient(name=name)

    assert p.name == name


@pytest.mark.parametrize(
    "values, expected, expected_exception",
    [
        ([[0, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0]], None),
        ([[1, 2], [0, 0], [0, 0]], [[0.5, 1], [0, 0], [0, 0]], None),
        ([[3, 2], [3, 6], [5, 9]], [[1, 2 / 3], [0.5, 1], [5 / 9, 1]], None),
        ([[3, 2], [-3, 6], [5, -9]], [[1, 2 / 3], [-0.5, 1], [5 / 9, -1]], None),
        ([[3, "hi"], [-3, 6], [5, -9]], None, ValueError),
    ]
)
def test_patient_normalise(values, expected, expected_exception):
    do_comparison = expected is not None and expected_exception is None
    do_exception = expected is None and expected_exception is not None

    assert do_comparison or do_exception

    with pytest.raises(expected_exception) if expected_exception is not None else nullcontext():
        result = patient_normalise(values)

        if do_comparison:
            npt.assert_allclose(result, expected, rtol=1e-05, atol=1e-08)
