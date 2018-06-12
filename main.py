# -*- coding: utf-8 -*-

import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html

from pandas_datareader import data as web
from datetime import datetime as dt
import plotly.plotly as py
import plotly.graph_objs as go

import numpy as np

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
from data_tools.data_splitter import DataSplitter
from models.linear_regression import LinearRegressionModel
from models.polynomial_model import PolynomialRegressionModel
from models.ridge_model import RidgeRegressionModel
from models.lasso_model import LassoRegressionModel

models_titles = {
    'BEST': 'Лучшее решение',
    'LINEAR': 'Линейная модель',
    'POLY': 'Полиномиальная модель',
    'RIDGE': 'Гребневая регрессия',
    'LASSO': 'Лассо'
}

models = [
    {
        'title': 'LINEAR',
        'class': LinearRegressionModel,
        'color': 'b'
    },
    {
        'title': 'POLY',
        'class': PolynomialRegressionModel,
        'color': 'r'
    },
    {
        'title': 'RIDGE',
        'class': RidgeRegressionModel,
        'color': 'y'
    },
    {
        'title': 'LASSO',
        'class': LassoRegressionModel,
        'color': 'm'
    },
]

# Считываем данные в DataFrame

data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)

x = data['TV']
y = data.sales
data_splitter = DataSplitter(x, y)
data_splitter.split()
train_data, test_data = [data_splitter.get_train_data(), data_splitter.get_test_data()]

models_instances = {}
min_err = 0
for mc in models:
    model = mc['class'](train_data)
    model.fit(test_data)
    predicted_data = model.predict(test_data['x'])
    err = mean_squared_error(predicted_data, test_data['y'])

    title = mc['title']
    models_instances[title] = {
        'model': model,
        'err': err
    }

    if err < min_err or min_err == 0:
        models_instances['BEST'] = {
            'model': model,
            'err': err,
            'title': models_titles[mc['title']]
        }

app = dash.Dash('Regression Analysis')

app.layout = html.Div([
    dcc.Input(
        style={'width': '500px'},
        id='input-text',
        placeholder='Введите url с данными',
        type='text'
    ),
    dcc.Input(
        style={'width': '500px', 'margin': '0 auto 20px auto'},
        id='my-id1',
        placeholder='Выберите файл с данными',
        type='file'
    ),
    dcc.Dropdown(
        id='my-dropdown',
        options=[
            {'label': models_titles['BEST'], 'value': 'BEST'},
            {'label': models_titles['LINEAR'], 'value': 'LINEAR'},
            {'label': models_titles['POLY'], 'value': 'POLY'},
            {'label': models_titles['RIDGE'], 'value': 'RIDGE'},
            {'label': models_titles['LASSO'], 'value': 'LASSO'}
        ],
        value='BEST'
    ),
    dcc.Checklist(
        options=[
            {'label': 'Произвести отбор признаков', 'value': 'Select'}
        ],
        values=[]
    ),
    dcc.Graph(id='my-graph'),
    html.Div([dcc.Slider(
        id='year--slider',
        min=5,
        max=25,
        value=20,
        step=5,
        marks={i: '{} %'.format(i) for i in np.arange(5, 30, 5)}
    )], style = {'margin-top': '10px'}),
    html.Div([html.P('Выберите размер контрольной выборки')], style={'width': '300', 'margin': '20px auto'}),
    html.Div([
        html.P(
            'Лучшая модель - {}'.format(models_instances['BEST']['title'])
        ),
        html.P(
            'Среднеквадратичная ошибка - {}'.format(models_instances['BEST']['err'])
        )
    ])
], style={'width': '750', 'margin': '10px auto'})


@app.callback(Output('my-graph', 'figure'), [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown_value):
    xx = range(int(min(test_data['x'])[0]), int(max(test_data['x'])[0]))
    xxx = map(lambda x: [x], xx)
    yy = models_instances[selected_dropdown_value]['model'].predict(xxx)
    # Create traces
    trace0 = go.Scatter(
        x=x,
        y=y,
        mode='markers',
        name='data'
    )
    trace2 = go.Scatter(
        x=xx,
        y=yy,
        mode='lines',
        name=models_titles[selected_dropdown_value],
        line=dict(width=3)
    )

    data = [trace0, trace2]
    return {
        'data': data,
        'layout': {
            'xaxis': {'title': 'feature'},
            'yaxis': {'title': 'answer'},
            'margin': {'l': 40, 'r': 0, 't': 20, 'b': 30}
        }
    }


app.css.append_css(
    {'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'})

if __name__ == '__main__':
    app.run_server()
