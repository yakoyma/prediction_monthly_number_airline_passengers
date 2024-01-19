"""
===============================================================================
Time Series Regression project: Prediction of the number of passengers of an
airline in a particular month using the SARIMAX model
===============================================================================

This file is organised as follows:
1. Data Analysis
2. Feature Engineering
3. Machine Learning
   3.1 Pmdarima
   3.2 Statsmodels
   3.3 Sktime
   3.4 PyCaret
"""
# Standard libraries
import random
import platform
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Other libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn
import pmdarima as pm
import statsmodels
import statsmodels.api as sm
import sktime
import pycaret

from sklearn.preprocessing import MinMaxScaler
from pmdarima.arima.utils import ndiffs
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.stats.diagnostic import acorr_ljungbox
from sktime.utils.plotting import plot_series
from sktime.transformations.series.lag import Lag
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.base import ForecastingHorizon
from sktime.transformations.series.detrend import Deseasonalizer, Detrender
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.forecasting.sarimax import SARIMAX
from pycaret.time_series import *
from functions import *

# Display versions of platforms and packages
print('\nPython: {}'.format(platform.python_version()))
print('Pandas: {}'.format(pd.__version__))
print('Seaborn: {}'.format(sns.__version__))
print('Scikit-learn: {}'.format(sklearn.__version__))
print('Pmdarima: {}'.format(pm.__version__))
print('Statsmodels: {}'.format(statsmodels.__version__))
print('Sktime: {}'.format(sktime.__version__))
print('PyCaret: {}'.format(pycaret.__version__))


# Constants
SEED = 0
PERIOD = 12
FOLDS = 3

# Set the random seed for reproducibility
random.seed(SEED)
np.random.seed(SEED)

# Set the maximum number of rows to display by Pandas
pd.set_option('display.max_rows', 200)

# Set the default Seaborn style
sns.set_style('whitegrid')



"""
===============================================================================
1. Data Analysis
===============================================================================
"""
# Loading the dataset
raw_dataset = load_dataset('air passengers.csv')

# Display the dataset's dimensions
print('\n\nDimensions of the dataset: {}'.format(raw_dataset.shape))

# Display the dataset's information
print('\nInformation about the dataset:')
print(raw_dataset.info())

# Display the head and the tail of the dataset
print(pd.concat([raw_dataset.head(), raw_dataset.tail()]))

# Description of the dataset
print('\nDescription of the dataset:')
print(round(raw_dataset.describe(include='all'), 0))

# The completion rate of the dataset
print(f'\nCompletion rate:\n{raw_dataset.count() / len(raw_dataset)*100}')

# Missing rate of the dataset
print(f'\nMissing rate:\n{raw_dataset.isna().mean() * 100}')

# Cleanse the dataset
dataset = raw_dataset.copy()

# Check for duplicated data
print(f'\nNumber of duplicated data: {dataset[dataset.duplicated()].shape[0]}')

dataset['Month'] = pd.to_datetime(dataset['Month'])
dataset = dataset.rename(
    columns={'#Passengers': 'Passengers', 'Month': 'Date'}).set_index('Date')
dataset.index = pd.PeriodIndex(dataset.index, freq='M')

# Display head and the tail of the dataset
print(pd.concat([dataset.head(), dataset.tail()]))

# Display the monthly number of passengers
fig, ax = plt.subplots(figsize=(12, 6))
plot_series(
    dataset,
    markers='.',
    x_label='Date',
    y_label='Passengers',
    ax=ax)
ax.set_title(f'Monthly number of passengers from '
             f'{dataset.index.min()} to {dataset.index.max()}')
plt.show()



"""
===============================================================================
2. Feature Engineering
===============================================================================
"""
# Tests to determine whether the dataset is stationary and/or invertible
print(f'\n\nStationarity test result: '
      f'{ArmaProcess(dataset).isstationary}')
print(f'Invertibility test result: '
      f'{ArmaProcess(dataset).isinvertible}')

# Split the dataset into training and test sets
train, test = temporal_train_test_split(
    dataset, test_size=int(0.25 * dataset.shape[0]))
print(f'\nTraining set shape: {train.shape}')
print(pd.concat([train.head(), train.tail()]))
print(f'\nTest set shape: {test.shape}')
print(pd.concat([test.head(), test.tail()]))

# Display training and test targets
fig, ax = plt.subplots(figsize=(12, 6))
plot_series(
    train,
    test,
    labels=['Training Target', 'Test Target'],
    markers=['.', '.'],
    x_label='Date',
    y_label='Passengers',
    ax=ax)
ax.set_title(f'Monthly number of passengers from '
             f'{train.index.min()} to {test.index.max()}')
plt.show()


# Transformation of the dataset to create an exogenous variable.
# Given the number of passengers in month x,
# the number of passengers is predicted for the following month x+1.
X = Lag([0, 1]).fit_transform(dataset).dropna()
X = X.rename(columns={
    'lag_0__Passengers': 'Target',
    'lag_1__Passengers': 'Exog'})

# Display head and the tail of the dataset
print(pd.concat([X.head(), X.tail()]))

# Display the target and exogenous variable
fig, ax = plt.subplots(figsize=(12, 6))
plot_series(
    X.Target,
    X.Exog,
    labels=['Target', 'Exogenous variable'],
    markers=['.', '.'],
    x_label='Date',
    y_label='Passengers',
    ax=ax)
ax.set_title(f'Monthly number of passengers from '
             f'{X.index.min()} to {X.index.max()}')
ax.grid(True)
plt.show()



"""
===============================================================================
3. Machine Learning
===============================================================================
"""
# 3.1 Pmdarima
# Determine whether to differentiate in order to make it stationary
# Estimate the number of times to differentiate using a KPSS test
print(f"\n\nEstimate the number of times to differentiate the data using "
      f"the KPSS test: {ndiffs(np.array(train), test='kpss')}")

# Instantiate and fit the model
auto_arima_model = pm.auto_arima(
    y=np.array(train),
    d=ndiffs(np.array(train), test='kpss'),
    m=PERIOD,
    seasonal=True,
    stationary=False,
    information_criterion='aic',
    n_jobs=-1,
    trend='t',
    method='lbfgs',
    trace=True,
    random_state=SEED)
print(f'\nSummary of the model:\n{auto_arima_model.summary()}')

# Make predictions
forecast = pd.Series(
    data=np.array(auto_arima_model.predict(test.shape[0])),
    index=np.array(test.index))
print(f'\nForecast:\n{forecast}')

# Display metrics
print('\nForecast target shape: ', np.array(forecast).shape)
print('Actual target shape: ', np.array(test).shape)
print('SMAPE: {:.3f}'.format(
    pm.metrics.smape(np.array(test), np.array(forecast).reshape(-1, 1))))
display_sktime_metrics(
    np.array(test),
    np.array(forecast).reshape(-1, 1),
    np.array(train),
    PERIOD)
display_sklearn_metrics(
    np.array(test), np.array(forecast).reshape(-1, 1), SEED)

# Display actual values of the training target
# and the test target compared to forecast
fig, ax = plt.subplots(figsize=(12, 6))
plot_series(
    train,
    test,
    forecast,
    labels=['Training Target', 'Test Target', 'Forecast'],
    markers=['.', '.', '.'],
    x_label='Date',
    y_label='Passengers',
    ax=ax)
ax.set_title(f'Training Target and Test Target vs. Forecast from '
             f'{train.index.min()} to {test.index.max()}')
plt.show()



# 3.2 Statsmodels
# Instantiate the model
model = sm.tsa.statespace.SARIMAX(
    endog=train,
    order=auto_arima_model.get_params()['order'],
    seasonal_order=auto_arima_model.get_params()['seasonal_order'],
    trend='t',
    enforce_stationarity=True,
    enforce_invertibility=True)

# Fit the model
res = model.fit(method='lbfgs')
print(f'\n\nSummary of the model:\n{res.summary()}')
print(f'\nParameters of the model:\n{res.params}')

# Display training metrics
print('\nResults pvalue: {:.3f}'.format(res.pvalues.mean()))
print('Results AIC: {:.3f}'.format(np.round(res.aic, 2)))
print('Results BIC: {:.3f}'.format(np.round(res.bic, 2)))
print('Results MSE: {:.3f}'.format(np.round(res.mse, 2)))
print('Results MAE: {:.3f}'.format(np.round(res.mae, 2)))
print('Ljung-Box stat: {:.3f}'.format(
    np.round(acorr_ljungbox(res.resid, period=PERIOD)['lb_stat'].mean(), 2)))
print('Ljung-Box pvalue: {:.3f}'.format(
    acorr_ljungbox(res.resid, period=PERIOD)['lb_pvalue'].mean()))

# Display diagnostics
res.plot_diagnostics(auto_ylims=True)
plt.show()

# Make forecasts
forecast = res.forecast(steps=test.shape[0], dynamic=False)
print(f'\nForecast:\n{forecast}')

# Display metrics for predictions
print('\nForecast target shape: ', np.array(forecast).shape)
print('Actual target shape: ', np.array(test).shape)
print('SMAPE: {:.3f}'.format(
    pm.metrics.smape(np.array(test).flatten(), np.array(forecast))))
display_sktime_metrics(
    np.array(test).flatten(),
    np.array(forecast),
    np.array(train),
    PERIOD)
display_sklearn_metrics(np.array(test).flatten(), np.array(forecast), SEED)

# Display actual values of the training target
# and the test target compared to forecast
fig, ax = plt.subplots(figsize=(12, 6))
plot_series(
    train,
    test,
    forecast,
    labels=['Training Target', 'Test Target', 'Forecast'],
    markers=['.', '.', '.'],
    x_label='Date',
    y_label='Passengers',
    ax=ax)
ax.set_title(f'Training Target and Test Target vs. Forecast from '
             f'{train.index.min()} to {test.index.max()}')
plt.show()



# 3.3 Sktime
# Instantiate the model
forecaster = TransformedTargetForecaster([
    ('Deseasonalizer', Deseasonalizer(sp=PERIOD)),
    ('Detrender', Detrender(PolynomialTrendForecaster(degree=1))),
    ('SARIMAX', SARIMAX(
        order=auto_arima_model.get_params()['order'],
        seasonal_order=auto_arima_model.get_params()['seasonal_order'],
        trend='t',
        enforce_stationarity=True,
        enforce_invertibility=True,
        random_state=SEED))])

# Fit the model and make predictions
forecast = forecaster.fit_predict(
    y=train, fh=ForecastingHorizon(test.index, is_relative=False))
print(f'\n\nForecast:\n{forecast}')

# Display metrics
print('\nForecast target shape: ', np.array(forecast).shape)
print('Actual target shape: ', np.array(test).shape)
print('SMAPE: {:.3f}'.format(
    pm.metrics.smape(np.array(test), np.array(forecast))))
display_sktime_metrics(
    np.array(test),
    np.array(forecast),
    np.array(train),
    PERIOD)
display_sklearn_metrics(np.array(test), np.array(forecast), SEED)

# Display actual values of the training target
# and the test target compared to forecast
fig, ax = plt.subplots(figsize=(12, 6))
plot_series(
    train,
    test,
    forecast,
    labels=['Training Target', 'Test Target', 'Forecast'],
    markers=['.', '.', '.'],
    x_label='Date',
    y_label='Passengers',
    ax=ax)
ax.set_title(f'Training Target and Test Target vs. Forecast from '
             f'{train.index.min()} to {test.index.max()}')
plt.show()



# 3.4 PyCaret
"""
PyCaret determines other effective models to solve the business problem.
This first selection is based on the basic hyperparameters of the models.
To select the final model, the best models must be compared on the basis of
their optimised hyperparameters.
"""
# Initialisation of the setup
s = setup(
    data=dataset,
    target='Passengers',
    scale_target='minmax',
    fold=FOLDS,
    fh=int(0.2 * dataset.shape[0]),
    n_jobs=-1,
    session_id=SEED)

# Selection of the best model by cross-validation using basic hyperparameters
best = compare_models(
    fold=FOLDS,
    round=3,
    cross_validation=True,
    n_select=1,
    sort='MASE')
print(f'\n\nClassification of models:\n{best}')

# Display diagnostics
plot_model(plot='diagnostics')

# Display forecast
plot_model(best, plot='forecast')


# The transformed dataset X with exogenous variable
# Initialisation of the setup
s = setup(
    data=X,
    target='Target',
    scale_target='minmax',
    scale_exogenous='minmax',
    fold=FOLDS,
    fh=int(0.2 * X.shape[0]),
    n_jobs=-1,
    session_id=SEED)

# Selection of the best model by cross-validation using basic hyperparameters
best = compare_models(
    fold=FOLDS,
    round=3,
    cross_validation=True,
    n_select=1,
    sort='MASE')
print(f'\n\nClassification of models:\n{best}')

# Display diagnostics
plot_model(plot='diagnostics')

# Display forecast
plot_model(best, plot='forecast')
