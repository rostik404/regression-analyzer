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
from models.base import BaseRegressionModel


class BasePolyRegressionModel(BaseRegressionModel):
    def fit(self, test_data):
        err = 0
        for degree in range(2, 10):
            est = self._make_estimator(degree)
            est.fit(self._tarin_data['x'], self._tarin_data['y'])

            new_err = mean_squared_error(
                test_data['y'], est.predict(test_data['x']))
            if new_err < err or err == 0:
                self._est = est
                self._degree = degree
            err = new_err

        self._fitted = True

    def _make_estimator(self, degree):
        pass

    def get_degree(self):
        return self._degree
