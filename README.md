Phase 1: Foundational Setup & Initial Model Training (Core)
Goal: Establish a baseline MLP model and a robust data pipeline.

Environment Setup
Install Python and the required libraries.

Tools: pip install tensorflow tensorflow-probability scikit-learn numpy pandas.

Verify GPU access: Use tf.config.list_physical_devices('GPU') to ensure TensorFlow detects your A4000 GPU and its CUDA cores.

Set up a Jupyter Notebook or Google Colab environment for interactive development.

Initial Data Loading and Preprocessing
Load the provided dataset (e.g., from a CSV file).

Separate inputs (Cell Size X, Cell Size Y/Z) from outputs (Pressure Drop, Surface Area, Mass).

Scaling: Use sklearn.preprocessing.StandardScaler to standardize both the inputs and outputs. This is critical for network convergence.

Split the provided data into a small training set and a validation set. A 90/10 split would be a good starting point. The bulk of your validation/test data will be generated in Phase 2.

Baseline MLP Model Training (Keras)
Action: Train a basic MLP model using Keras on the initial provided training data. This will serve as your "before" model to demonstrate the impact of targeted sampling.

Tools: tf.keras.Sequential for building the network. tf.keras.optimizers.Adam and mean_squared_error for loss. Use model.fit() with the validation data for early stopping.

Phase 2: Targeted Sampling and Model Refinement (Core)
Goal: Demonstrate the efficiency and impact of your data augmentation strategy.

Analyze Baseline Residuals
Action: Use the baseline model to make predictions on the validation set. Calculate the residuals (errors) and visualize them. Create a scatter plot or a 2D heatmap of the input parameter space, with a color scale indicating the magnitude of the model's prediction error. Identify the regions with the highest errors.

Tools: matplotlib or seaborn for visualization.

Strategic Data Point Generation
Action: Select a small number of new input points (e.g., 5-10) to generate.

Strategy: Combine two approaches:

Target the high-residual regions identified in the previous step.

Use Latin Hypercube Sampling (scipy.stats.qmc.LatinHypercube) to generate a small, space-filling set of points across the entire parameter space to ensure broad coverage.

nTop Automation: Write a Python script using nTop Automate to run simulations for these new input parameters and save the results.

Final Model Training
Action: Combine the original provided data with the new strategically generated data. Retrain your MLP model on this enhanced dataset. This will be your final, core surrogate model.

Demonstrate Improvement: Present a "before" vs. "after" comparison showing the reduction in residuals/error, especially in the previously problematic regions. Quantify the "error reduction per additional data point" as planned.

Phase 3: Inverse Design Optimization (Core)
Goal: Use the trained MLP to find the optimal design parameters.

Define Objective Function
Action: Create a Python function that takes the raw input parameters (Cell Size X, Cell Size Y/Z), feeds them through your trained and scaled MLP model, and returns the Performance metric. Remember to invert the scaling on the outputs before calculating performance.

Tools: Your Keras model, numpy, and scikit-learn scalers.

Gradient-Based Inverse Design
Action: Use TensorFlow's automatic differentiation to perform gradient ascent on the input parameters.

Tools:

tf.Variable: Wrap your initial Cell Size X and Cell Size Y/Z in tf.Variable to make them "trainable."

tf.GradientTape: Use tf.GradientTape to record the operations of your forward pass.

tf.optimizers.Adam: Use the Adam optimizer to update the input variables in the direction that maximizes the performance (i.e., minimizes -1 * Performance).

Multiple Starting Points: Run this optimization multiple times with different random starting points within the parameter boundaries (7-15mm) to find multiple local optima.

Verification
Take the best optimum found by your surrogate and run a final nTop simulation to get the "true" performance score.

Phase 4: Stretch Goal - Uncertainty & Robustness (Optional)
Goal: Enhance the model's trustworthiness and provide a robustness scheme for designers.

BNN Model Implementation
Action: Replace the standard Keras Dense layers with tfp.layers.DenseReparameterization to build a Bayesian Neural Network.

Loss Function: Your loss function will now have two parts:

The standard reconstruction loss (mean_squared_error).

The KL divergence loss, which is a regularization term that encourages the weight distributions to stay close to their priors. This is automatically handled by the DenseReparameterization layers.

Uncertainty-Informed Robustness Scheme
Action: For each optimum found in Phase 3, use your BNN to get not just a single prediction, but a distribution of predictions (e.g., by running 100 forward passes with the BNN). This gives you the mean and standard deviation (uncertainty) at that point.

Defining Robustness: Create a function that takes an optimum point and a "sensitivity limit" (e.g., ±5%) specified by the user.

This function would evaluate the model's predicted performance at multiple points within this tolerance box around the optimum.

A "robust" design would be one where the performance remains relatively high and stable within that box, indicated by low standard deviation. A "sensitive" design would be one where the performance drops off steeply, even if the peak value is higher.

Visualization: Plot the performance score (and its uncertainty) as a function of the user's specified tolerance for a few different local optima. This visually shows which designs are more robust to manufacturing variation.

Phase 5: Stretch Goal - Low-Fidelity Surrogate (Optional)
Goal: Demonstrate the broader application of your methodology for multi-fidelity design.

Low-Fidelity Data Generation
Action: Generate a very small, cheap dataset using a simplified, lower-fidelity model. For example, if there's a simple analytical approximation for a lattice's properties, use that. Or, if nTop can be run in a lower-resolution or faster mode, use that to generate the data.

Low-Fidelity Surrogate Training
Action: Train a small, simple MLP model on this low-fidelity data.

Demonstrate Use Case
Show that this model is extremely fast for exploring the entire design space. Identify several promising regions based on its predictions. These regions are where you would then run your expensive, high-fidelity nTop simulations, validating your efficiency claim.

Inverse Design with Low-Fidelity Model: Run inverse design on this model to find "low-fidelity optima." Then, use these low-fidelity optima as intelligent starting points for the high-fidelity inverse design in Phase 3.

Phase 6: Presentation and Report
Goal: Synthesize your findings into a compelling story.

Slide 1: Problem & Approach: Briefly introduce the problem and your "physics-informed" approach (data efficiency, BNNs for robustness).

Slide 2: Data Efficiency: Show the "before" vs. "after" error maps, the small number of new data points you generated, and the "error reduction per data point" metric.

Slide 3: Inverse Design: Show the contour plot of the performance metric and highlight the multiple optima found by your gradient descent method.

Slide 4: Uncertainty & Robustness: Introduce the BNN and its ability to provide uncertainty. Present your robustness scheme and the visualization showing how performance changes with sensitivity limits, demonstrating how a designer can use your tool.

Slide 5: Broader Application: Discuss the low-fidelity modeling strategy and how your methodology can be applied to other complex design problems (e.g., robust design, real-time control).

Slide 6: Conclusion: Summarize the benefits: high accuracy, high efficiency, and trustworthiness through uncertainty quantification.