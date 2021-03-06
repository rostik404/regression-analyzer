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
from models.base_poly import BasePolyRegressionModel


class PolynomialRegressionModel(BasePolyRegressionModel):
    def _make_estimator(self, degree):
        return make_pipeline(PolynomialFeatures(self._degree), LinearRegression())
