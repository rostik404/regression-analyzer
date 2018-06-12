# -*- coding: utf-8 -*-

import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from data_tools.data_adapter import DataAdapter
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV


class BaseRegressionModel(object):
    """expects data object in format {x: [...], y: [...]}"""

    def __init__(self, train_data):
        self._tarin_data = train_data
        self._degree = 1
        self._est = self._make_estimator(self._degree)
        self._fitted = False

    def _make_estimator(self, degree):
        pass

    def fit(self, test_data):
        self._est.fit(self._tarin_data['x'], self._tarin_data['y'])
        self._fitted = True

    def get_degree(self):
        pass

    def predict(self, control_data):
        if not self._fitted:
            raise Exception(
                'You should fit the model before making prediction')

        return self._est.predict(control_data)

    def set_train_data(self, data):
        self._tarin_data = data
