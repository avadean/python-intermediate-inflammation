"""Tests for the Patient model."""

from inflammation.models import Patient
import numpy as np
import numpy.testing as npt

from inflammation.models import patient_normalise

def test_create_patient():

    name = 'Alice'
    p = Patient(name=name)

    assert p.name == name

def test_patient_normalise():
    test_input = np.array([[3, 2],
                           [3, 6],
                           [5, 9]])
    test_result = np.array([[1, 2 / 3],
                           [0.5, 1],
                           [5 / 9, 1]])

    npt.assert_allclose(patient_normalise(test_input), test_result)
