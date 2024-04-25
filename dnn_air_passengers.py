"""
===============================================================================
Time Series Regression project: Prediction of the number of passengers of an
airline in a particular month using the Deep Neural Network (DNN)
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
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import sweetviz as sv
import sklearn
import sktime
import tensorflow as tf
import keras_tuner

from sweetviz import analyze
from sklearn.preprocessing import MinMaxScaler
from sktime.utils.plotting import plot_series
from sktime.transformations.series.lag import Lag
from tensorflow.keras.callbacks import EarlyStopping
from keras_tuner.tuners import BayesianOptimization
from functions import *

# Display versions of platforms and packages
print('\nPython: {}'.format(platform.python_version()))
print('NumPy: {}'.format(np.__version__))
print('Matplotlib: {}'.format(matplotlib.__version__))
print('Pandas: {}'.format(pd.__version__))
print('Sweetviz: {}'.format(sv.__version__))
print('Scikit-learn: {}'.format(sklearn.__version__))
print('Sktime: {}'.format(sktime.__version__))
print('TensorFlow: {}'.format(tf.__version__))
print('KerasTuner: {}'.format(keras_tuner.__version__))


# Constant
SEED = 0

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

# Description of the dataset
print('\nDescription of the dataset:')
print(round(raw_dataset.describe(include='all'), 0))

# Display the head and the tail of the dataset
print(pd.concat([raw_dataset.head(), raw_dataset.tail()]))

# Time Series raw dataset report
report = analyze(source=raw_dataset, target_feat='#Passengers')
report.show_html('raw_dataset_report.html')

# Cleanse the dataset
dataset = raw_dataset.copy()
dataset['Month'] = pd.to_datetime(dataset['Month'])
dataset = dataset.rename(
    columns={'#Passengers': 'Passengers', 'Month': 'Date'}).set_index('Date')
dataset.index = pd.PeriodIndex(dataset.index, freq='M')

# Display the dataset's dimensions
print('\nDimensions of the dataset: {}'.format(dataset.shape))

# Display the dataset's information
print('\nInformation about the dataset:')
print(dataset.info())

# Description of the dataset
print('\nDescription of the dataset:')
print(round(dataset.describe(include='all'), 0))

# Display head and the tail of the dataset
print(pd.concat([dataset.head(), dataset.tail()]))

# Time Series dataset report
report = analyze(source=dataset, target_feat='Passengers')
report.show_html('dataset_report.html')

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
# Transformation of the dataset to create exogenous variables.
# Creation of 12 lags (features)
X = Lag([i for i in range(13)]).fit_transform(dataset).dropna()
X = X.rename(columns={'lag_0__Passengers': 'Target'})

# Display the dataset's dimensions
print('\n\nDimensions of the dataset: {}'.format(X.shape))

# Display the dataset's information
print('\nInformation about the dataset:')
print(X.info())

# Description of the dataset
print('\nDescription of the dataset:')
print(round(X.describe(include='all'), 0))

# Display head and the tail of the dataset
print(pd.concat([X.head(), X.tail()]))

# Time Series X dataset report
report = analyze(source=X, target_feat='Target')
report.show_html('X_dataset_report.html')

# Display the target
fig, ax = plt.subplots(figsize=(12, 6))
plot_series(
    X.Target,
    labels=['Target'],
    markers=['.'],
    x_label='Date',
    y_label='Passengers',
    ax=ax)
ax.set_title(f'Monthly number of passengers from '
             f'{X.index.min()} to {X.index.max()}')
ax.grid(True)
plt.show()

# Split the dataset into training and test sets
train_set = X.iloc[0:int(0.7 * X.shape[0]),]
test_set = X.iloc[train_set.shape[0]:X.shape[0],]

# Display head and the tail of the datasets
print(f'\nTraining set shape: {train_set.shape}')
print(pd.concat([train_set.head(), train_set.tail()]))
print(f'\nTest set shape: {test_set.shape}')
print(pd.concat([test_set.head(), test_set.tail()]))

# Time Series train set report
report = analyze(source=train_set, target_feat='Target')
report.show_html('train_set_report.html')

# Time Series test set report
report = analyze(source=test_set, target_feat='Target')
report.show_html('test_set_report.html')

# Display training and test targets
fig, ax = plt.subplots(figsize=(12, 6))
plot_series(
    train_set.Target,
    test_set.Target,
    labels=['Training Target', 'Test Target'],
    markers=['.', '.'],
    x_label='Date',
    y_label='Passengers',
    ax=ax)
ax.set_title(f'Monthly number of passengers from '
             f'{train_set.index.min()} to {test_set.index.max()}')
ax.grid(True)
plt.show()

# Standardisation
# Scale values between the range of 0 and 1
transformer = MinMaxScaler()
y_train = transformer.fit_transform(np.array(train_set.Target).reshape(-1, 1))
y_test = transformer.transform(np.array(test_set.Target).reshape(-1, 1))

scaler = MinMaxScaler()
X_train = scaler.fit_transform(np.array(train_set.drop(['Target'], axis=1)))
X_test = scaler.transform(np.array(test_set.drop(['Target'], axis=1)))
print(f'\nX_train shape: {X_train.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_test shape: {y_test.shape}')



"""
===============================================================================
3. Machine Learning
===============================================================================
"""
# Train and evaluate the DNN model
# Build the model
# Instantiate the tuner
tuner = BayesianOptimization(
    hypermodel=build_dnn_model,
    objective='val_loss',
    max_trials=10,
    seed=SEED)

# Create the callback that stops training the model
# when the loss stops improving
early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    verbose=0,
    patience=3)

# Train and find the best model
tuner.search(
    x=X_train,
    y=y_train,
    epochs=50,
    batch_size=64,
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
print(f'\ny_pred shape: {y_pred.shape}')

# Evaluate the model
scores = model.evaluate(X_test, y_test, verbose=0)
for metric, score in zip(['Loss (MSE)', 'MSE', 'MAE'], scores):
    print(f'{metric}: {score:.3f}')

# Display metrics for predictions
display_tensorflow_metrics(y_test, y_pred)
display_sklearn_metrics(y_test, y_pred, SEED)

# Display predictions
fig, ax = plt.subplots(figsize=(8, 5))
plot_series(
    pd.Series(data=y_test.flatten(), index=test_set.index),
    pd.Series(data=y_pred.flatten(), index=test_set.index),
    labels=['Test', 'Predictions'],
    markers=['.', '.'],
    x_label='Date',
    ax=ax)
ax.set_title(f'Actual vs. Predictions from '
             f'{test_set.index.min()} to {test_set.index.max()}')
ax.grid(True)
plt.show()

# Denormalise the predicted target
pred_target = transformer.inverse_transform(y_pred)
print('\nPredicted target shape: ', pred_target.shape)
print('Predicted target:\n', pred_target.flatten())
print('\nActual target shape: ', np.array(test_set.Target).shape)
print('Actual target:\n', np.array(test_set.Target))

# Display metrics for actual values
display_tensorflow_metrics(
    np.array(test_set.Target), pred_target.flatten())
display_sklearn_metrics(
    np.array(test_set.Target), pred_target.flatten(), SEED)

# Display actual values of the training target
# and the test target compared to predictions
fig, ax = plt.subplots(figsize=(12, 6))
plot_series(
    train_set.Target,
    test_set.Target,
    pd.Series(data=pred_target.flatten(), index=test_set.index),
    labels=['Training Target', 'Test Target', 'Predictions'],
    markers=['.', '.', '.'],
    x_label='Date',
    y_label='Passengers',
    ax=ax)
ax.set_title(f'Training Target and Test Target vs. Predictions from '
             f'{train_set.index.min()} to {test_set.index.max()}')
ax.grid(True)
plt.show()
