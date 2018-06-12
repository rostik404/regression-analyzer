# -*- coding: utf-8 -*-

import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.cross_validation import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from data_tools.data_adapter import DataAdapter
from models.base import BaseRegressionModel


class LinearRegressionModel(BaseRegressionModel):
    def _make_estimator(self, degree):
        return LinearRegression()

    def fit(self, test_data):
        self._est.fit(self._tarin_data['x'], self._tarin_data['y'])
        self._fitted = True
