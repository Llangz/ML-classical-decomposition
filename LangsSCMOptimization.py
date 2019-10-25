import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('decomposition.csv')
data.columns = ['date', 'y']
data = data.set_index(pd.to_datetime(data.ix[:, 0])).drop('date', axis=1)
print(data.head())

plt.figure(figsize=(14,10))
plt.plot(data['y'])
plt.show()

#Time series decomposition
#Splitting data
train = data.loc[data.index < '2017-01-01']
test = data.loc[data.index > '2017-01-01']

#Isolating Seasonality
#Monthly cycle
df_year = pd.DataFrame({'y': train.y.resample('M').sum()})
df_year['mva'] = df_year.y.rolling(center=True, window=12).mean()

plt.figure(figsize=(12, 8))
plt.plot(df_year['y'], label='Monthly Average')
plt.legend(loc='best')
plt.show()

#Seasonal Ratio
df_year['sr'] = df_year['y'] / df_year['mva']

#Adding month numbers
df_year['month'] = df_year.index.month

#for example, extract all January data
df_ = df_year.loc[df_year['month']==1]
print(df_)

#To get the average for each month
df_ = df_year.groupby('month').agg({'sr': 'mean'})
df_.reset_index(inplace=True)
df_.columns = ['month', 'usi']

#Combining with main dataframe

df_year = pd.merge(df_year, df_, on='month', right_index=True).sort_index()

df_year['asi'] = df_['usi'].mean() * df_year['usi']

#We can substitute the S′ values in for St to get the de-seasonalized values (yt−s):
df_year['y_t-s'] = df_year['y'] /df_year['asi']

#Plotting the deseasonalized values against actual values

plt.figure(figsize=(12, 8))
plt.plot(df_year['y'], label="$y$")
plt.plot(df_year['y_t-s'], label='$y_{t-s}$')
plt.legend(loc='best')
plt.show()

#Trend
from sklearn.linear_model import LinearRegression

lm = LinearRegression(normalize=False, fit_intercept=True)
y_t_s = np.atleast_2d(df_year['y_t-s'].values).T
x = np.atleast_2d(np.linspace(0, len(df_year) - 1, len(df_year))).T
lm.fit(x, y_t_s)
df_year['trend'] = lm.predict(x)

# Plot actual data, de-seasonalized data, and the trend
plt.figure(figsize=(12, 8))
plt.plot(df_year['y'], label='$y$')
plt.plot(df_year['y_t-s'], label='$y_t{t-s}$')
plt.plot(df_year['trend'], label="$T'$")
plt.legend(loc='best')
plt.show()

#Noise
df_year['noise'] = (df_year['y'] / (df_year['asi'] * df_year['trend'])).mean()
df_year.head()

test_year = pd.DataFrame({"y": test.y.resample("M").sum()})
test_year['month'] = test_year.index.month

#Index for trend regression
x_test = np.linspace(len(df_year), len(df_year) + len(test_year) - 1, len(test_year)).reshape(-1, 1)
df_test = pd.merge(test_year, df_year[['month', 'asi', 'noise']], on='month', right_index=True).sort_index().drop_duplicates()
df_test['trend'] = lm.predict(x_test)
df_test['forecast'] = df_test['asi'] * df_test['noise'] * df_test['trend']
print(df_test)

#Plotting the forecast
plt.figure(figsize=(12, 8))
plt.plot(df_year['y'], label='Train ($y_t$)')
plt.plot(df_test['y'], label='Test ($y_t$)')
plt.plot(df_test['forecast'], label='Forecast ($\hat{y_t}$)')
plt.legend(loc='best')
plt.title("Classical Decomposition and Multiplicative Model Forecast")
plt.show()

#Evaluation/ Forecast metrics
evaluation = df_test.copy()
evaluation['error'] = evaluation['y'] - evaluation['forecast']
evaluation.insert(0, 'series', 1) # insert value to groupby
evaluation.groupby('series').agg({
        'y' : 'sum',
        'forecast' : 'sum',
        'error': {
            'total_error' : 'sum',
            'percentage_error' : lambda x: 100 * np.sum(x) / np.sum(evaluation['y']),
            'mae': lambda x: np.mean(np.abs(x)),
            'rmse': lambda x: np.sqrt(np.mean(x ** 2)),
            'mape': lambda x: 100 * np.sum(np.abs(x)) / np.sum(evaluation['y'])
        }}).apply(np.round, axis=1)

#Score
from sklearn.externals import joblib
joblib.dump(lm, 'classical_decomposition_regression_model')

cd = joblib.load('classical_decomposition_regression_model')











