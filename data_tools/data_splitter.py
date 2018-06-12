# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
from data_adapter import DataAdapter

class DataSplitter(object):
    def __init__(self, x, y, test_size=0.8):
        self.set_data(x, y)
        self._test_size = test_size
        self._train_data = {'x': [], 'y': []}
        self._test_data = {'x': [], 'y': []}

    def set_data(self, x, y):
        self._data = DataAdapter(x, y).get_data()

    def split(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self._data['x'],
            self._data['y'],
            test_size=self._test_size
        )

        self._train_data = {'x': X_train, 'y': y_train}
        self._test_data = {'x': X_test, 'y': y_test}

    def get_test_data(self):
        return self._test_data

    def get_train_data(self):
        return self._test_data
