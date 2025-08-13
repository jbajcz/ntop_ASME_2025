# %%
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

print("TensorFlow version:", tf.__version__)
print("KerasTuner version:", kt.__version__)

# %%
# Phase 1: Initial Data Loading and Preprocessing
# 1. Load dataset
possible_paths = [
    'nTop/nTop ASME Hackathon Data.csv',
    'Generated Data.csv',
    './Generated Data.csv',
    './nTop/nTop ASME Hackathon Data.csv'
]

data = None
for file_path in possible_paths:
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        print("First 5 rows of the dataset:")
        print(data.head())
        print("\nDataset Info:")
        data.info()
        break
    except FileNotFoundError:
        continue

if data is None:
    print("Error: Could not find the data file in any of these locations:")
    for path in possible_paths:
        print(f"- {path}")
    print("Please ensure one of these files exists.")
    exit(1)

# %%
# 2. Separate features and targets
features = ['X Cell Size', 'YZ Cell Size', 'Velocity Inlet']
targets = ['PressureDrop', 'Surface Area', 'Mass']

X = data[features]
y = data[targets]

# 3. Standardize inputs & outputs
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)

print("Features and targets separated and scaled.")

# %%
# 4. Split data
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_scaled, test_size=0.1, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")

# %%
# Phase 2: Baseline MLP Model Training (Keras) with Keras Tuner

def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Input(shape=(X_train.shape[1],)))
    
    # Tune the number of layers
    for i in range(hp.Int('num_layers', 2, 5)):
        # Tune the number of units in each layer
        model.add(layers.Dense(units=hp.Int(f'units_{i}', min_value=32, max_value=512, step=32),
                               activation='relu'))
    
    model.add(layers.Dense(y_train.shape[1])) # Output layer
    
    # Tune the learning rate
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss='mean_squared_error',
                  metrics=['mean_absolute_error'])
    
    return model

# %%
tuner = kt.Hyperband(build_model,
                     objective='val_loss',
                     max_epochs=50,
                     factor=3,
                     directory='keras_tuner_dir',
                     project_name='asme_hackathon')

# Create a callback to stop training early after reaching a certain value for the validation loss.
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Start the search
tuner.search(X_train, y_train, epochs=100, validation_data=(X_val, y_val), callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of layers is {best_hps.get('num_layers')} 
and the optimal learning rate for the optimizer is {best_hps.get('learning_rate')}.
""")

# %%
# Build the model with the optimal hyperparameters and train it on the data
model = tuner.hypermodel.build(best_hps)
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), callbacks=[stop_early])

val_loss = history.history['val_loss']
best_epoch = val_loss.index(min(val_loss)) + 1
print(f'Best epoch: {best_epoch}')

# %%
hypermodel = tuner.hypermodel.build(best_hps)

# Retrain the model
hypermodel.fit(X_train, y_train, epochs=best_epoch)

eval_result = hypermodel.evaluate(X_val, y_val)
print(f"[test loss, test accuracy]: {eval_result}")

# %%
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

# %%
# Identify high-error regions
# We'll use the mean squared error of the scaled residuals to identify the points with the highest error.
mse_per_point = np.mean(np.square(residuals_scaled), axis=1)
high_error_indices = np.argsort(mse_per_point)[-10:] # Get top 10 high-error points

print("Top 10 validation points with highest prediction error:")
X_val_orig = scaler_X.inverse_transform(X_val)
high_error_inputs = X_val_orig[high_error_indices]
print(pd.DataFrame(high_error_inputs, columns=features))

# Visualize high-error regions
plt.figure(figsize=(10, 8))
sc = plt.scatter(X_val_orig[:, 0], X_val_orig[:, 1], c=mse_per_point, cmap='viridis', alpha=0.7)
plt.colorbar(sc, label='Mean Squared Error')
plt.scatter(high_error_inputs[:, 0], high_error_inputs[:, 1], color='red', s=100, edgecolor='black', label='High-Error Points')
plt.xlabel(features[0])
plt.ylabel(features[1])
plt.title('Validation Set Prediction Error')
plt.legend()
plt.show()

# %%
# 2. Strategic Data Point Generation
import subprocess
import json
import os
import csv

# Define paths and file names
exePath = r"C:/Program Files/nTopology/nTopology/nTopCL.exe"
nTopFilePath = r"nTop/nTop_ASME_Hackathon_HEX.ntop"
Input_File_Name = "input.json"
Output_File_Name = "output.json"
new_csv_file_path = "nTop/augmented_data.csv"

# Check if nTopCL executable exists
if not os.path.exists(exePath):
    print(f"nTopCL.exe not found at {exePath}")
    print("Skipping data generation.")
else:
    # Input variables in JSON structure
    Inputs_JSON = {
        "description": "",
        "inputs": [
            {"description": "", "name": "Cell Size X", "type": "real", "units": "mm", "value": 25.0},
            {"description": "", "name": "Cell Size Y/Z", "type": "real", "units": "mm", "value": 25.0},
            {"description": "", "name": "Inlet Velocity", "type": "real", "units": "mm*s^-1", "value": 3000.0}
        ],
        "title": "Simple Heat Exchanger"
    }

    # nTopCL arguments
    Arguments = [exePath, "-j", Input_File_Name, "-o", Output_File_Name, nTopFilePath]

    print("Generating new data points for high-error regions...")
    # Use the high_error_inputs identified previously
    for i, params in enumerate(high_error_inputs):
        # Set the input parameters in the JSON
        Inputs_JSON['inputs'][0]['value'] = float(params[0]) # X Cell Size
        Inputs_JSON['inputs'][1]['value'] = float(params[1]) # YZ Cell Size
        Inputs_JSON['inputs'][2]['value'] = float(params[2]) # Inlet Velocity
        
        # Create input.json file
        with open(Input_File_Name, 'w') as outfile:
            json.dump(Inputs_JSON, outfile, indent=4)

        # nTopCL call
        print(f"Running simulation for point {i+1}/{len(high_error_inputs)}: {params}")
        process = subprocess.Popen(Arguments, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            print(f"Error running nTopCL for point {params}:")
            print(stderr)
            continue

        # Read the JSON from the output file
        with open(Output_File_Name, 'r') as output_file:
            output_data = json.load(output_file)
            # The actual data is nested
            sim_results = output_data[0]['value']['jsonObject']

        # On the first iteration, write headers to the new CSV
        if i == 0:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(new_csv_file_path), exist_ok=True)
            with open(new_csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(sim_results.keys())
        
        # Append the new data
        with open(new_csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(sim_results.values())

    print(f"\nNew data generation complete. Data saved to '{new_csv_file_path}'")

# %%
# 3. Final Model Training
# Load the new data and merge with the original dataset
try:
    new_data = pd.read_csv(new_csv_file_path)
    print(f"Successfully loaded {len(new_data)} new data points.")
    
    # The new data from the simulation contains both inputs and outputs.
    # We need to reorder columns to match the original `data` DataFrame before concatenation.
    
    # Let's construct the new data correctly. The simulation output in `new_data`
    # needs to be combined with the inputs that generated it.
    new_inputs_df = pd.DataFrame(high_error_inputs, columns=features)
    
    # The `new_data` from the CSV has different column names than the original data.
    # Let's select the target columns from the new_data and create the augmented dataframe.
    new_outputs_df = new_data[targets]
    
    # Let's create a new dataframe for the augmented data with the correct columns
    # Reset index to ensure proper alignment
    augmented_df = pd.concat([new_inputs_df.reset_index(drop=True), new_outputs_df.reset_index(drop=True)], axis=1)
    
    # Combine with original data
    combined_data = pd.concat([data, augmented_df], ignore_index=True)
    print("Original and new data combined.")
    print(f"Total dataset size: {len(combined_data)} points.")

    # -- Re-preprocess the entire dataset --
    X_combined = combined_data[features]
    y_combined = combined_data[targets]

    # Standardize inputs & outputs
    scaler_X_final = StandardScaler()
    X_scaled_final = scaler_X_final.fit_transform(X_combined)

    scaler_y_final = StandardScaler()
    y_scaled_final = scaler_y_final.fit_transform(y_combined)

    # Split data
    X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
        X_scaled_final, y_scaled_final, test_size=0.1, random_state=42
    )
    
    print("Combined data has been preprocessed for final training.")

except FileNotFoundError:
    print(f"'{new_csv_file_path}' not found. Skipping final model training.")
    combined_data = None
except Exception as e:
    print(f"An error occurred: {e}")
    combined_data = None

# %%
# Retrain the MLP on the combined dataset
if combined_data is not None:
    print("\nRetraining model on the augmented dataset...")
    
    # Set up the tuner for the final training
    tuner_final = kt.Hyperband(build_model,
                               objective='val_loss',
                               max_epochs=50,
                               factor=3,
                               directory='keras_tuner_dir',
                               project_name='asme_hackathon_final',
                               overwrite=True) # Overwrite previous results in this directory

    # Start the new hyperparameter search
    tuner_final.search(X_train_final, y_train_final, 
                       epochs=100, 
                       validation_data=(X_val_final, y_val_final), 
                       callbacks=[stop_early])

    # Get the optimal hyperparameters from the new search
    best_hps_final = tuner_final.get_best_hyperparameters(num_trials=1)[0]

    print(f"""
    The final hyperparameter search is complete. 
    The optimal number of layers is {best_hps_final.get('num_layers')} 
    and the optimal learning rate for the optimizer is {best_hps_final.get('learning_rate')}.
    """)

    # Build the model with the new optimal hyperparameters
    final_model = tuner_final.hypermodel.build(best_hps_final)
    
    # Find the best epoch from the tuning history
    best_epoch_final = tuner_final.oracle.get_best_trials(num_trials=1)[0].best_step

    # Retrain the final model on the full augmented training data for the optimal number of epochs
    history_final = final_model.fit(X_train_final, y_train_final, 
                                    epochs=best_epoch_final,
                                    validation_data=(X_val_final, y_val_final),
                                    callbacks=[stop_early],
                                    verbose=1)
    
    print("\nFinal model training complete.")
    
    # Evaluate the final model
    final_eval_result = final_model.evaluate(X_val_final, y_val_final)
    print(f"\nFinal model evaluation on augmented validation set:")
    print(f"[test loss, test accuracy]: {final_eval_result}")

# %%
# Compare Before vs. After Residual Maps
if combined_data is not None:
    print("\nAnalyzing residuals of the final model...")
    
    # Predict on the validation set with the new model
    y_pred_scaled_final = final_model.predict(X_val_final)
    
    # Calculate new residuals
    residuals_scaled_final = y_val_final - y_pred_scaled_final
    mse_per_point_final = np.mean(np.square(residuals_scaled_final), axis=1)
    
    # Inverse transform for visualization
    X_val_orig_final = scaler_X_final.inverse_transform(X_val_final)

    # Plot the new error map
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(X_val_orig_final[:, 0], X_val_orig_final[:, 1], c=mse_per_point_final, cmap='viridis', alpha=0.7)
    plt.colorbar(sc, label='Mean Squared Error (After Augmentation)')
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.title('Validation Set Prediction Error (After Augmentation)')
    plt.show()

    # You can also compare the MAE before and after
    y_pred_final = scaler_y_final.inverse_transform(y_pred_scaled_final)
    y_val_orig_final = scaler_y_final.inverse_transform(y_val_final)
    residuals_final = y_val_orig_final - y_pred_final
    mae_per_target_final = np.mean(np.abs(residuals_final), axis=0)

    print("\nMean Absolute Error Comparison:")
    print("Target         | Before      | After")
    print("---------------------------------------")
    for i, target in enumerate(targets):
        print(f"{target:<15} | {mae_per_target[i]:<11.4f} | {mae_per_target_final[i]:.4f}")


