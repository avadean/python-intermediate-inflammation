from unittest.mock import Mock

import numpy as np
import numpy.testing as npt

from inflammation.compute_data import CSVDataSource, analyse_data


def test_analyse_data_mock_source():

    data_source = Mock()

    data_source.load_inflammation_data.return_value = np.array([[[1, 2, 3],
                                                                [4, 5, 6],
                                                                [7, 8, 9],
                                                                [10, 11, 12],
                                                                [13, 14, 15]]])

    analyse_data(data_source)


def test_analyse_data_regression():
    data_source = CSVDataSource("mini_data/")

    result = analyse_data(data_source)
    result = result.get("standard deviation by day")

    expected = np.array([6.72061505, 5.43650214, 4.4032816])

    npt.assert_allclose(result, expected, rtol=1E-3, atol=1E-5)
