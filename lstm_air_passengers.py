"""
===============================================================================
Time Series Regression project: Prediction of the number of passengers of an
airline in a particular month using the Long Short-Term Memory (LSTM) network
===============================================================================

This file is organised as follows:
1. Data Analysis
2. Feature Engineering
3. Machine Learning
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
import sklearn
import tensorflow as tf
import keras_tuner
import sktime

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from sktime.utils.plotting import plot_series
from functions import *

# Display versions of platforms and packages
print('\nPython: {}'.format(platform.python_version()))
print('Pandas: {}'.format(pd.__version__))
print('Scikit-learn: {}'.format(sklearn.__version__))
print('TensorFlow: {}'.format(tf.__version__))
print('KerasTuner: {}'.format(keras_tuner.__version__))
print('Sktime: {}'.format(sktime.__version__))


# Constants
SEED = 0
TIMESTEP = 1
BATCH_SIZE = 16
EPOCHS = 100
PATIENCE = 5

# Set the random seed for reproducibility
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Set the maximum number of rows to display by Pandas
pd.set_option('display.max_rows', 200)



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
ax.grid(True)
plt.show()



"""
===============================================================================
2. Feature Engineering
===============================================================================
"""
# Transformation of the dataset to create an exogenous variable.
# Given the number of passengers in month x,
# the number of passengers is predicted for the following month x+1.
X = generate_data_batches(
    np.array(dataset).reshape(-1, 1),
    np.array(dataset.index).reshape(-1, 1),
    TIMESTEP)

# Display head and the tail of the dataset
print(pd.concat([X.head(), X.tail()]))

# Display the target and exogenous variable
fig, ax = plt.subplots(figsize=(12, 6))
plot_series(
    pd.Series(data=X.Target, index=X.index),
    pd.Series(data=X.Exog, index=X.index),
    labels=['Target', 'Exogenous variable'],
    markers=['.', '.'],
    x_label='Date',
    y_label='Passengers',
    ax=ax)
ax.set_title(f'Monthly number of passengers from '
             f'{X.index.min()} to {X.index.max()}')
ax.grid(True)
plt.show()

# Split the dataset into training and test sets
train = X.iloc[0:int(0.8 * X.shape[0]),]
test = X.iloc[train.shape[0]:X.shape[0],]

# Display head and the tail of the datasets
print(f'\nTraining set shape: {train.shape}')
print(pd.concat([train.head(), train.tail()]))
print(f'\nTest set shape: {test.shape}')
print(pd.concat([test.head(), test.tail()]))

# Display training and test targets
fig, ax = plt.subplots(figsize=(12, 6))
plot_series(
    pd.Series(data=train.Target, index=train.index),
    pd.Series(data=test.Target, index=test.index),
    labels=['Training Target', 'Test Target'],
    markers=['.', '.'],
    x_label='Date',
    y_label='Passengers',
    ax=ax)
ax.set_title(f'Monthly number of passengers from '
             f'{train.index.min()} to {test.index.max()}')
ax.grid(True)
plt.show()

# Standardisation
# Scale values between the range of 0 and 1
exog_scaler = MinMaxScaler()
X_train = exog_scaler.fit_transform(np.array(train.Exog).reshape(-1, 1))
X_test = exog_scaler.transform(np.array(test.Exog).reshape(-1, 1))

scaler = MinMaxScaler()
y_train = scaler.fit_transform(np.array(train.Target).reshape(-1, 1))
y_test = scaler.transform(np.array(test.Target).reshape(-1, 1))

# Reshape exogenous variable three-dimensional for the LSTM model
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
print(f'\nX_train shape: {X_train.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_test shape: {y_test.shape}')



"""
===============================================================================
3. Machine Learning
===============================================================================
"""
# Train and evaluate the LSTM model
# Build the model
# Instantiate the tuner
tuner = keras_tuner.RandomSearch(
    hypermodel=build_lstm_model,
    objective='val_loss',
    max_trials=5,
    seed=SEED)

# Create the callback that stops training the model
# when the loss stops improving
early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    mode='min',
    verbose=0,
    patience=PATIENCE)

# Train and find the best model
tuner.search(
    x=X_train,
    y=y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    callbacks=[early_stopping_callback])

# Get the best model with optimised hyperparameters
model = tuner.get_best_models(num_models=1)[0]
print('\n\nOptimal hyperparameters:')
print(tuner.get_best_hyperparameters()[0].values)
print('\nSummary of the best model with optimised hyperparameters:')
print(model.summary())

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
scores = model.evaluate(X_test, y_test, verbose=0)
for metric, score in zip(['Loss (MSE)', 'MSE', 'MAE'], scores):
    print(f'{metric}: {score:.3f}')

# Denormalise the predicted target
print(f'\ny_pred shape: {y_pred.shape}')
pred_target = scaler.inverse_transform(y_pred).astype(int)
print('Predicted target shape: ', pred_target.shape)
print('Predicted target:\n', pred_target.flatten())
print('\nActual target shape: ', np.array(test.Target).shape)
print('Actual target:\n', np.array(test.Target))

# Display metrics for predictions
display_tensorflow_metrics(y_test, y_pred)
display_sklearn_metrics(y_test, y_pred, SEED)

# Display predictions
fig, ax = plt.subplots(figsize=(8, 5))
plot_series(
    pd.Series(data=y_test.flatten(), index=test.index),
    pd.Series(data=y_pred.flatten(), index=test.index),
    labels=['Test', 'Predictions'],
    markers=['.', '.'],
    x_label='Date',
    ax=ax)
ax.set_title(f'Actual vs. Predictions from '
             f'{test.index.min()} to {test.index.max()}')
ax.grid(True)
plt.show()

# Display metrics for actual values
display_tensorflow_metrics(
    np.array(test.Target), pred_target.flatten())
display_sklearn_metrics(
    np.array(test.Target), pred_target.flatten(), SEED)

# Display actual values of the training target
# and the test target compared to predictions
fig, ax = plt.subplots(figsize=(12, 6))
plot_series(
    train.Target,
    test.Target,
    pd.Series(data=pred_target.flatten(), index=test.index),
    labels=['Training Target', 'Test Target', 'Predictions'],
    markers=['.', '.', '.'],
    x_label='Date',
    y_label='Passengers',
    ax=ax)
ax.set_title(f'Training Target and Test Target vs. Predictions from '
             f'{train.index.min()} to {test.index.max()}')
ax.grid(True)
plt.show()
