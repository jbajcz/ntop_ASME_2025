# ntop_ASME_2025

## Environment Setup
1. **Install Python and required libraries**  
   ```bash
   pip install tensorflow tensorflow-probability scikit-learn numpy pandas
    ```

2. **Verify GPU access**

   ```python
   tf.config.list_physical_devices('GPU')
   ```

   Ensure TensorFlow detects your A4000 GPU and its CUDA cores.

3. **Set up development environment**

   * Use **Jupyter Notebook** or **Google Colab** for interactive development.

---

## Initial Data Loading and Preprocessing

1. **Load dataset** (e.g., CSV file).
2. **Separate features and targets**:

   * Inputs: `Cell Size X`, `Cell Size Y/Z`
   * Outputs: `Pressure Drop`, `Surface Area`, `Mass`
3. **Standardize inputs & outputs**

   * Use `sklearn.preprocessing.StandardScaler` for both.
   * This is crucial for network convergence.
4. **Split data**

   * 90% training, 10% validation.

---

## Baseline MLP Model Training (Keras)

1. Train a **basic MLP** using Keras on the training data.
2. Use:

   * `tf.keras.Sequential` for network structure.
   * `tf.keras.optimizers.Adam` optimizer.
   * `mean_squared_error` loss.
3. Train with:

   ```python
   model.fit(..., validation_data=(...), callbacks=[...])
   ```

   * Include **early stopping**.

---

## Phase 2: Targeted Sampling & Model Refinement (Core)

### Goal

Demonstrate the efficiency and impact of your data augmentation strategy.

### Analyze Baseline Residuals

1. Predict on validation set.
2. Compute residuals (errors).
3. Visualize:

   * Scatter plot or 2D heatmap.
   * Color scale = prediction error magnitude.
4. Identify **high-error regions**.

   * Tools: `matplotlib`, `seaborn`.

### Strategic Data Point Generation

1. Select **5–10** new input points.
2. Strategy:

   * Focus on **high-residual regions**.
   * Use `scipy.stats.qmc.LatinHypercube` for space-filling coverage.
3. Run nTop Automate simulations for these points.
4. Save results.

### Final Model Training

1. Merge **original** + **new** data.
2. Retrain MLP = **final core surrogate model**.
3. Compare:

   * Before vs. after residual maps.
   * **Error reduction per additional data point**.

---

## Phase 3: Inverse Design Optimization (Core)

### Goal

Use the trained MLP to find optimal design parameters.

### Define Objective Function

1. Takes raw input parameters (`Cell Size X`, `Cell Size Y/Z`).
2. Feeds through **scaled MLP**.
3. Inverts scaling on outputs.
4. Returns **Performance metric**.

### Gradient-Based Inverse Design

1. Use **TensorFlow automatic differentiation**:

   * `tf.Variable` for trainable inputs.
   * `tf.GradientTape` for tracking.
   * `tf.optimizers.Adam` for updates.
2. Run optimization from multiple random start points (7–15 mm bounds).
3. Select multiple local optima.

### Verification

* Run **nTop simulation** for best optimum.
* Compare to predicted performance.

---

## Phase 4: Stretch Goal – Uncertainty & Robustness (Optional)

### Goal

Increase trustworthiness with robustness analysis.

### BNN Model Implementation

1. Replace `Dense` with:

   ```python
   tfp.layers.DenseReparameterization
   ```
2. Loss = `mean_squared_error` + KL divergence.

### Uncertainty-Informed Robustness Scheme

1. For each optimum:

   * Run multiple forward passes.
   * Get mean & standard deviation.
2. Evaluate performance within **±5% tolerance**.
3. Define **robust design** = high mean, low std in tolerance box.
4. Visualization:

   * Performance vs. uncertainty plots.

---

## Phase 5: Stretch Goal – Low-Fidelity Surrogate (Optional)

### Goal

Show multi-fidelity design workflow.

### Low-Fidelity Data Generation

* Use low-res nTop simulations or analytical approximations.

### Low-Fidelity Surrogate Training

* Train small MLP.

### Demonstrate Use Case

1. Explore design space cheaply.
2. Identify promising regions.
3. Use these as starting points for **high-fidelity Phase 3** optimization.

---

## Phase 6: Presentation & Report

### Goal

Tell a compelling story.

**Slides:**

1. **Problem & Approach** – Physics-informed strategy.
2. **Data Efficiency** – Before vs. after error maps, error reduction per data point.
3. **Inverse Design** – Contour plots & multiple optima.
4. **Uncertainty & Robustness** – BNN + robustness scheme.
5. **Broader Application** – Low-fidelity modeling.
6. **Conclusion** – Accuracy, efficiency, trustworthiness.

---
