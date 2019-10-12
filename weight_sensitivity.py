import pandas as pd
import numpy as np
import sys
from statsmodels.tsa.vector_ar.var_model import VAR
import sys
import warnings
warnings.filterwarnings("ignore")

# #############################################
# Input parameters
# #############################################
SENSITIVITY_RATIO = 0.3


def perform_analysis(key=''):
    # Read CSV From SQL Table
    df = pd.read_csv('https://www.yquantify.com/csv/daily.csv?key=' + key)

    ##############################################
    # DATA EXPLORATION
    ##############################################
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    columns = ['sleep', 'exercise', 'calories', 'weight']
    df = df.copy()
    df = df[columns]

    ##############################################
    # Data manupulation for NAN value
    ##############################################
    nan_columns = df.isnull().any()
    nan_true = [i for i, col_i in enumerate(nan_columns) if col_i == True]

    for nan_i in nan_true:
        column_name = nan_columns.index[nan_i]
        s = pd.Series(df[column_name])
        df[column_name] = s.interpolate(method='nearest')

    ##############################################
    #  ARIMA
    #  (Auto-Regressive Integrated Moving Average)
    ##############################################

    train = df[:-1]

    # # Make final predictions.
    model = VAR(endog=train)
    model_fit = model.fit()
    yhat = model_fit.forecast(model_fit.y, steps=2)

    # ##############################################
    # # Sensitivity
    # ##############################################

    def sensitivity(df, col_name, ratio, percentage=0.9):
        df_sen[col_name].iloc[-2] = df_sen[col_name].iloc[-2] * ratio

        train = df_sen[:-1]

        model_sen = VAR(endog=train)
        model_sen_fit = model_sen.fit()

        # Make prediction on validation.
        yhat_sen_cal = model_sen_fit.forecast(model_sen_fit.y, steps=2)
        return yhat_sen_cal[:, 3][-1]

    col_names_sensitivity = {
        'sleep': 1+SENSITIVITY_RATIO,
        'exercise': 1+SENSITIVITY_RATIO,
        'calories': 1-SENSITIVITY_RATIO}

    weight_pred = []
    weight_diff = []
    for col_i, direction in col_names_sensitivity.items():
        df_sen = df.copy()
        res = sensitivity(df_sen, col_i, direction, percentage=0.1)
        weight_pred.append(res)
        weight_diff.append(yhat[1, 3] - res)

    # #############################################
    # Final result
    # #############################################
    df_sen = pd.DataFrame()
    df_sen['attr'] = col_names_sensitivity.keys()
    df_sen['weight_pred'] = weight_pred
    df_sen['sensitivity'] = np.array(weight_diff) * -1

    return df_sen.to_dict()
