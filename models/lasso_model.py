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
from models.base_poly import BasePolyRegressionModel


class LassoRegressionModel(BasePolyRegressionModel):
    def _make_estimator(self, degree):
        return make_pipeline(PolynomialFeatures(degree), LassoCV(alphas=[0.1, 0.2, 0.3, 0.4]))
