"""Module containing mechanism for calculating standard deviation between datasets.
"""

import glob
import os
import numpy as np

from inflammation import models, views


class CSVDataSource:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_inflammation_data(self):
        data_file_paths = glob.glob(os.path.join(self.data_dir, 'inflammation*.csv'))

        if not len(data_file_paths):
            raise ValueError(f"No inflammation data CSV files found in path {self.data_dir}")

        return list(map(models.load_csv, data_file_paths))


def daily_mean(data):
    return map(models.daily_mean, data)


def daily_standard_deviation(means_by_day_matrix):
    return np.std(means_by_day_matrix, axis=0)


def analyse_data(data_source: CSVDataSource):
    """Calculates the standard deviation by day between datasets.

    Gets all the inflammation data from CSV files within a directory,
    works out the mean inflammation value for each day across all datasets,
    then plots the graphs of standard deviation of these means."""
    data = data_source.load_inflammation_data()

    means_by_day_matrix = np.stack(list(daily_mean(data)))

    graph_data = {
        'standard deviation by day': daily_standard_deviation(means_by_day_matrix),
    }

    views.visualize(graph_data)

    return graph_data
