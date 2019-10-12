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

# Read CSV From SQL Table
df = pd.read_csv(
    'https://www.yquantify.com/csv/daily.csv?key=0400ECBA82BD7EAFFBAACA6950938CC06AF39770A57F7DE49810097BA45FF7AD734D208F5B1169BB6236ABCFCC060B9D')

##############################################
# DATA EXPLORATION
##############################################
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

columns = ['sleep', 'exercise', 'calories', 'weight']
df = df.copy()
df = df[columns]

# Cast to
for column in columns:
    df[column] = df[column].astype(float)

##############################################
# Data manupulation for NAN value
##############################################
cols = df.columns

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

X = df.drop(['weight'], axis=1)
y = df['weight']

# Fit the model

model = VAR(endog=df)
model_fit = model.fit()

# # Make prediction on validation.
# prediction = model_fit.forecast(model_fit.y, steps=1)

# # Make final predictions.
# model = VAR(endog=train)
# model_fit = model.fit()
# yhat = model_fit.forecast(model_fit.y, steps=2)

# ##############################################
# # Sensitivity
# ##############################################


# def sensitivity(df, col_name, ratio, percentage=0.9):
#     df_sen[col_name].iloc[-2] = df_sen[col_name].iloc[-2] * ratio

#     train = df_sen[:-1]

#     model_sen = VAR(endog=train)
#     model_sen_fit = model_sen.fit()

#     # Make prediction on validation.
#     yhat_sen_cal = model_sen_fit.forecast(model_sen_fit.y, steps=2)
#     return yhat_sen_cal[:, 3][-1]


# col_names_sensitivity = {
#     'sleep': 1+SENSITIVITY_RATIO,
#     'exercise': 1+SENSITIVITY_RATIO,
#     'calories': 1-SENSITIVITY_RATIO}

# weight_pred = []
# weight_diff = []
# for col_i, direction in col_names_sensitivity.items():
#     df_sen = df.copy()
#     res = sensitivity(df_sen, col_i, direction, percentage=0.1)
#     weight_pred.append(res)
#     weight_diff.append(yhat[1, 3] - res)

# # #############################################
# # Final result
# # #############################################
# df_sen = pd.DataFrame()
# df_sen['attr'] = col_names_sensitivity.keys()
# df_sen['weight_pred'] = weight_pred
# df_sen['sensitivity'] = np.array(weight_diff) * -1

# print(df_sen.to_json())
# sys.stdout.flush()
