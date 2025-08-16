# Heat Exchanger Optimization Project Documentation
## Overview
This project uses a machine learning model to perform inverse design for a heat exchanger. The goal is to find the optimal physical parameters (cell sizes and inlet velocity) that maximize the heat exchanger's surface area while meeting specific constraints on mass, pressure drop, and average flow velocity.

The project consists of three main components:

1. `main.py`: A Python script that uses the nTopology command-line interface (`nTopCL.exe`) to generate a dataset of heat exchanger designs and their performance metrics.

2. `train.ipynb`: A Jupyter notebook that trains a neural network model to predict heat exchanger performance based on the input parameters from the dataset. It uses Keras and KerasTuner to find the best model architecture.

3. `inverse_design.ipynb`: A Jupyter notebook that uses the trained neural network model and `scipy.optimize.minimize` to find the optimal input parameters for a new heat exchanger design based on a set of defined constraints. It also includes a validation step using `nTopCL.exe` to check the accuracy of the optimized result.

## Project Methodology
Our approach to this inverse design problem follows a standard surrogate modeling and optimization workflow:

1. Data Generation: Using the provided `nTop` file, we run a full factorial experiment to generate a comprehensive dataset. The main.py script automates this process by systematically varying the input parameters (`X Cell Size`, `Y/Z Cell Size`, and `Inlet Velocity`) and collecting the corresponding output metrics (`Pressure Drop`, `Avg Velocity`, `Surface Area`, and `Mass`) from nTopology. This step creates the high-quality simulation data needed to train our surrogate model.

2. Surrogate Model Training: The `train.ipynb` notebook is used to develop a neural network that acts as a surrogate for the nTopology simulations. We preprocess the generated data by separating features and targets and then scaling them to improve model performance. Using KerasTuner, we explore different neural network architectures and hyperparameters to find the optimal configuration. This process allows us to create a model that can predict the output metrics with high accuracy but in a fraction of the time it takes for a full simulation.

3. Inverse Design Optimization: The `inverse_design.ipynb` notebook leverages the trained surrogate model to solve the inverse design problem. Instead of running slow simulations, the notebook uses the fast-predicting neural network to evaluate the objective and constraint functions. We employ a multi-start, gradient-based optimization algorithm (Sequential Least Squares Programming, or SLSQP) from `scipy.optimize.minimize`. This method efficiently searches the design space to find the input parameters that maximize the surface area while satisfying the given constraints.

4. Validation: To ensure the trustworthiness of our inverse design results, we perform a validation step. The optimal parameters found by the optimization are passed back into the `nTop` file, and a final, full simulation is run. The outputs from this simulation are then compared to the predictions from our surrogate model to verify the accuracy of our approach.

`main.py`
This script is responsible for generating the initial dataset used for training the machine learning model.

### Purpose
The script iterates through a defined range of input parameters—`"Cell Size X"`, `"Cell Size Y/Z"`, and `"Inlet Velocity"`—and runs a series of nTopology simulations. For each simulation, it records the output metrics, such as `"PressureDrop"`, `"AvgVelocity"`, `"Surface Area"`, and `"Mass"`, and saves them to a CSV file named `Generated Data.csv`. This dataset is crucial for training the neural network.

### Key Components
- `Inputs_JSON`: A Python dictionary that defines the structure and default values of the input parameters for the nTopology script.

- `Arguments`: A list that constructs the command-line arguments for `nTopCL.exe`, specifying the input JSON file, the output JSON file, and the nTopology notebook file (`.ntop`).

- Nested Loops: The script uses three nested `for` loops to systematically vary the input parameters within a specified range and number of steps (e.g., `np.linspace(10, 25, 5)`).

- `subprocess.Popen`: This is the core function call that executes nTopCL.exe with the configured arguments, running the simulation for each combination of input parameters.

- CSV Output: The script writes the headers of the output data to a CSV file on the first iteration and then appends the data from subsequent iterations.

`train.ipynb`
This notebook details the process of training a surrogate model (a neural network) to predict heat exchanger performance.

### Purpose
The primary goal of this notebook is to replace the time-consuming nTopology simulations with a fast-predicting machine learning model. This model can then be used efficiently for optimization tasks.

### Steps
1. Data Loading and Preprocessing: The notebook loads the `Generated Data.csv` file, separates the input features from the output targets, and scales both the features and targets using `MinMaxScaler`. This scaling step is important for improving the performance and stability of the neural network training.

2. Data Splitting: The data is split into a training set and a validation set to evaluate the model's performance on unseen data.

3. Model Building: It defines a function, `build_model`, that constructs a neural network using Keras. This function is designed for a hyperparameter tuning framework to allow KerasTuner to search for the best number of layers, units per layer, activation functions, and learning rates.

4. Hyperparameter Tuning: KerasTuner's `Hyperband` algorithm is used to efficiently search for the optimal set of hyperparameters for the neural network.

5. Model Training: The final model is trained on the full training dataset using the best hyperparameters found during the tuning process. It includes an `EarlyStopping` callback to prevent overfitting.

6. Model Evaluation and Saving: After training, the model's performance is evaluated by calculating the Mean Absolute Error (MAE) for each output target. Finally, the trained model and its associated scalers are saved as pickle files (`trained_model.pkl`, `scaler_X.pkl`, `scaler_y.pkl`) for use in the inverse design notebook.

`inverse_design.ipynb`
This notebook uses the trained machine learning model to solve the inverse design problem.

### Purpose
The core function of this notebook is to find the optimal heat exchanger design (input parameters) that satisfies specific performance constraints and maximizes a given objective, such as surface area. It also includes a critical validation step to verify the model's prediction with an actual nTopology simulation.

### Key Components
- Model Loading: The notebook loads the saved neural network and scalers from the pickle files, enabling it to make fast predictions.

- Objective and Constraint Functions:

- Objective Function: The optimization aims to maximize `Surface Area`, so the objective function is defined as the negative of the surface area to be minimized by the solver.

- Constraint Functions: These functions define the design's performance boundaries. The constraints are written to return a non-negative value if they are satisfied (e.g., `125 - mass > 0`). The constraints are: `Mass < 125 g`, `Pressure Drop < 8000 Pa`, and `Avg Velocity > 520 mm/s`.

- Optimization with `scipy.optimize.minimize`: The `minimize` function from the `scipy.optimize` library is used to perform a multi-start optimization. It finds the set of input parameters that minimizes the objective function while adhering to the defined constraints.

- Results and Validation: The notebook prints the optimal input parameters and the predicted outputs from the surrogate model. It then performs a final validation step: it uses the same `nTopCL.exe` call from `main.py` to run a simulation with the newly optimized parameters and compares the results to the model's predictions.