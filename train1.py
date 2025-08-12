import os

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth for each GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

print("TensorFlow version:", tf.__version__)
print("KerasTuner version:", kt.__version__)

# Phase 1: Initial Data Loading and Preprocessing
# 1. Load dataset
file_path = 'nTop/Generated Data.csv'
try:
    data = pd.read_csv(file_path)
    print("Data loaded successfully.")
    print("First 5 rows of the dataset:")
    print(data.head())
    print("\nDataset Info:")
    data.info()
except FileNotFoundError:
    print(f"Error: The file was not found at {file_path}")
    print("Please make sure the CSV file is in the 'nTop' directory.")

# 2. Separate features and targets
features = ['X Cell Size', 'YZ Cell Size', 'Velocity Inlet']
targets = ['PressureDrop', 'Surface Area', 'Mass']

X = data[features]
y = data[targets]

# 3. Standardize inputs & outputs using MinMaxScaler
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y)

print("Features and targets separated and scaled with MinMaxScaler.")

# 4. Split data
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_scaled, test_size=0.1, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")

# Phase 2: Baseline MLP Model Training (Keras) with Keras Tuner


def build_model(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(X_train.shape[1],)))

    # Tune the number of layers
    for i in range(hp.Int('num_layers', 2, 6)):
        # Tune the number of units in each layer
        model.add(tf.keras.layers.Dense(units=hp.Int(f'units_{i}', min_value=32, max_value=1024, step=64),
                                         activation=hp.Choice(f'activation_{i}', ['relu', 'tanh', 'elu'])))
        # Add dropout for regularization
        model.add(tf.keras.layers.Dropout(hp.Float(f'dropout_{i}', 0.0, 0.5, step=0.1)))

    model.add(tf.keras.layers.Dense(y_train.shape[1]))  # Output layer

    # Tune the learning rate
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 5e-5])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss='mean_squared_error',
                  metrics=['mean_absolute_error'])

    return model


tuner = kt.Hyperband(build_model,
                     objective='val_loss',
                     max_epochs=100,
                     factor=3,
                     directory='keras_tuner_dir',
                     project_name='asme_hackathon')

# Create a callback to stop training early after reaching a certain value for the validation loss.
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

# Start the search
tuner.search(X_train, y_train, epochs=100, validation_data=(X_val, y_val), callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of layers is {best_hps.get('num_layers')}
and the optimal learning rate for the optimizer is {best_hps.get('learning_rate')}.
""")

# Build the model with the optimal hyperparameters and train it on the data
model = tuner.hypermodel.build(best_hps)
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), callbacks=[stop_early])

val_loss = history.history['val_loss']
best_epoch = val_loss.index(min(val_loss)) + 1
print(f'Best epoch: {best_epoch}')

hypermodel = tuner.hypermodel.build(best_hps)

# Retrain the model
hypermodel.fit(X_train, y_train, epochs=best_epoch)

eval_result = hypermodel.evaluate(X_val, y_val)
print(f"[test loss, test accuracy]: {eval_result}")

# Phase 2: Targeted Sampling & Model Refinement
# 1. Analyze Baseline Residuals
y_pred_scaled = hypermodel.predict(X_val)
residuals_scaled = y_val - y_pred_scaled

# Inverse transform to get original scale
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_val_orig = scaler_y.inverse_transform(y_val)
residuals = y_val_orig - y_pred

# Calculate Mean Absolute Error for each target
mae_per_target = np.mean(np.abs(residuals), axis=0)
print("Mean Absolute Error for each target:")
for i, target in enumerate(targets):
    print(f"{target}: {mae_per_target[i]:.4f}")

# Visualize residuals
plt.figure(figsize=(15, 5))
for i, target in enumerate(targets):
    plt.subplot(1, len(targets), i + 1)
    sns.histplot(residuals[:, i], kde=True)
    plt.title(f'Residuals for {target}')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Identify high-error regions
# We'll use the mean squared error of the scaled residuals to identify the points with the highest error.
mse_per_point = np.mean(np.square(residuals_scaled), axis=1)
# Get top 100 high-error points, or all if less than 100 are available
num_points = min(100, len(X_val))
high_error_indices = np.argsort(mse_per_point)[-10:]

print(f"Top {num_points} validation points with highest prediction error:")
X_val_orig = scaler_X.inverse_transform(X_val)
high_error_inputs = X_val_orig[high_error_indices]
print(pd.DataFrame(high_error_inputs, columns=features))

# Visualize high-error regions
plt.figure(figsize=(10, 8))
sc = plt.scatter(X_val_orig[:, 0], X_val_orig[:, 1], c=mse_per_point, cmap='viridis', alpha=0.7)
plt.colorbar(sc, label='Mean Squared Error')
plt.scatter(high_error_inputs[:, 0], high_error_inputs[:, 1], color='red', s=100, edgecolor='black',
            label='High-Error Points')
plt.xlabel(features[0])
plt.ylabel(features[1])
plt.title('Validation Set Prediction Error')
plt.legend()
plt.show()