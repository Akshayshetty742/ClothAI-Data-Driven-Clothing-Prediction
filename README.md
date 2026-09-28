# ClothAI — Data-Driven Clothing Prediction

A machine-learning mini-project that learns a mapping from a 56-dimensional human-pose representation to a high-dimensional clothing deformation representation.

---

## 1. Project Overview

**ClothAI — Data-Driven Clothing Prediction** is a research-oriented machine-learning prototype for predicting clothing deformation from human-pose features.

The project uses a PCA-based dimensional-reduction pipeline followed by a fully connected neural network.

The complete workflow is:

```text
56-Dimensional Human Pose
        ↓
Input Standardization
        ↓
PCA Dimensionality Reduction
        ↓
128 PCA Components
        ↓
Fully Connected Neural Network
        ↓
Predicted PCA Coefficients
        ↓
Inverse PCA
        ↓
6,510-Dimensional Clothing Representation
        ↓
Evaluation and Visualization
```

---

## 2. Objective

The objective of this project is to investigate whether a neural network can learn the relationship between human pose features and a high-dimensional clothing deformation representation.

The model receives a 56-dimensional pose feature vector as input and predicts a compact PCA representation of the clothing deformation.

The predicted PCA representation is then transformed back into the original 6,510-dimensional space using inverse PCA.

---

## 3. Problem Statement

Clothing deformation is a high-dimensional problem because changes in human pose can affect many points in a clothing representation.

Directly predicting thousands of output values can make the learning problem unnecessarily complex.

To address this, the project uses PCA to compress the high-dimensional clothing representation into a smaller number of meaningful components.

A neural network is then trained to predict these PCA coefficients from the human-pose input.

---

## 4. Input and Output

### Input

The model receives:

```text
56 numerical pose features
```

The 56 features represent:

```text
14 joints × 4 quaternion parameters
```

Therefore:

```text
14 × 4 = 56 features
```

### Output

The original clothing representation contains:

```text
6,510 values
```

PCA reduces this representation to:

```text
128 PCA components
```

The neural network predicts these 128 PCA coefficients.

After prediction, inverse PCA reconstructs the:

```text
6,510-dimensional clothing representation
```

---

## 5. Methodology

The project follows the following machine-learning pipeline.

### Step 1 — Input Pose Representation

A 56-dimensional human-pose vector is provided as the input.

```text
14 joints × 4 quaternion parameters
                ↓
        56-dimensional vector
```

### Step 2 — Input Standardization

The 56 input features are standardized using a trained input scaler.

The scaler ensures that the input features are represented on a suitable numerical scale before being passed to the neural network.

The trained scaler is stored as:

```text
models/x_scaler.joblib
```

### Step 3 — PCA Transformation

The original 6,510-dimensional clothing representation is transformed using Principal Component Analysis (PCA).

PCA reduces the dimensionality of the output representation while retaining the major variation present in the data.

The project uses:

```text
6,510-dimensional representation
                ↓
              PCA
                ↓
       128 PCA components
```

The trained PCA model is stored as:

```text
models/pca.joblib
```

### Step 4 — Neural Network

A fully connected neural network is trained to learn the mapping:

```text
56 pose features
       ↓
Neural Network
       ↓
128 PCA coefficients
```

The trained model is stored as:

```text
models/mlp.pt
```

### Step 5 — Predicted PCA Coefficients

During prediction, the neural network produces:

```text
128 predicted PCA coefficients
```

These coefficients represent the predicted clothing deformation in the reduced PCA space.

### Step 6 — Inverse PCA

The predicted PCA coefficients are transformed back into the original representation using inverse PCA.

```text
128 predicted PCA coefficients
                ↓
           Inverse PCA
                ↓
6,510-dimensional clothing representation
```

### Step 7 — Evaluation and Visualization

The predicted output is evaluated using regression metrics.

The project also generates:

- Training and validation loss curves
- PCA explained-variance visualization
- Prediction visualizations
- Numerical evaluation metrics

---

## 6. PCA

Principal Component Analysis (PCA) is used to reduce the dimensionality of the clothing representation.

The original output contains:

```text
6,510 values
```

PCA reduces this representation to:

```text
128 components
```

The experiment achieved:

```text
PCA explained variance = 0.9998849
```

This corresponds to approximately:

```text
99.98849% explained variance
```

The PCA explained-variance plot is saved as:

```text
results/pca_variance.png
```

---

## 7. Neural Network Architecture

The neural network learns the mapping from the 56-dimensional pose representation to the 128-dimensional PCA representation.

The architecture is represented as:

```text
Input Layer
56 features
    ↓
Dense Layer
128 neurons
    ↓
ReLU
    ↓
Dense Layer
256 neurons
    ↓
ReLU
    ↓
Output Layer
128 PCA coefficients
```

The trained model is stored as:

```text
models/mlp.pt
```

---

## 8. Training Configuration

The training process uses:

- Loss function: Mean Squared Error (MSE)
- Optimizer: Adam
- Early stopping
- Validation monitoring
- Training/validation/test split

The current experiment uses:

```text
Training samples   : 1,800
Validation samples : 600
Testing samples    : 600
```

Total:

```text
3,000 samples
```

---

## 9. Early Stopping

Early stopping is used during neural-network training.

The model monitors validation performance and stops training when the validation loss no longer improves for the configured patience period.

The training log showed:

```text
Early stopping.
Training complete.
```

---

## 10. Experimental Results

The current experiment produced the following neural-network test metrics:

| Metric | Value |
|---|---:|
| Neural Network MSE | 0.00150401 |
| Neural Network RMSE | 0.03878156 |
| Neural Network MAE | 0.03073971 |

The project also includes a linear-regression baseline.

### Linear Regression Baseline

```text
Linear Regression MSE = 0.00109090
```

The baseline is included to provide a reference point for evaluating the neural-network model.

These results are specific to the current experiment and dataset.

---

## 11. Training Results

The training process produced:

```text
Train/Validation/Test
1800 / 600 / 600
```

The final training log also reported:

```text
PCA explained variance: 0.9998849
```

The training and validation losses are saved in:

```text
results/loss_history.csv
```

The corresponding visualization is saved as:

```text
results/loss_curve.png
```

---

## 12. Results Visualization

### Training and Validation Loss

The loss curve shows the change in training and validation MSE across training epochs.

File:

```text
results/loss_curve.png
```

The curve helps visualize the learning behaviour of the neural network.

### PCA Explained Variance

The PCA variance plot shows the cumulative amount of variance explained as the number of PCA components increases.

File:

```text
results/pca_variance.png
```

The final experiment achieved approximately:

```text
99.98849% explained variance
```

with the selected PCA representation.

---

## 13. Streamlit Application

The project includes an interactive Streamlit dashboard.

The dashboard provides a research-style interface for:

- Live prediction
- Results
- Methodology
- Dataset and scope
- Model evaluation
- Visualization

The application can be started using:

```bash
streamlit run app.py
```

---

## 14. Live Prediction

The Live Prediction section allows the user to provide a 56-dimensional pose vector.

The application supports predefined pose configurations such as:

```text
Zero pose
Random pose — seed 7
Random pose — seed 42
```

It also supports a custom pose vector.

The custom input must contain exactly:

```text
56 comma-separated numerical values
```

The application validates the number of values before running the prediction.

---

## 15. Prediction Pipeline

When the user clicks **Run prediction**, the application performs the following steps:

```text
User Input
    ↓
Parse 56 numerical values
    ↓
Validate input size
    ↓
Input scaling
    ↓
Neural Network inference
    ↓
128 PCA coefficients
    ↓
Inverse PCA
    ↓
6,510 output values
    ↓
Prediction visualization
    ↓
Prediction summary
```

The application also provides an option to download the prediction output as a CSV file.

---

## 16. Prediction Summary

The dashboard provides summary information for the predicted clothing representation, including:

- Number of output values
- Number of vertices
- Mean offset
- Standard deviation
- Predicted deformation profile

The current output contains:

```text
6,510 output values
```

The dashboard visualizes the predicted deformation profile against the output-coordinate index.

---

## 17. Project Structure

```text
data_driven_clothing_project/
│
├── data/
│
├── models/
│   ├── mlp.pt
│   ├── pca.joblib
│   └── x_scaler.joblib
│
├── results/
│   ├── loss_curve.png
│   ├── loss_history.csv
│   ├── metrics.txt
│   └── pca_variance.png
│
├── src/
│
├── app.py
├── README.md
├── REPORT_TEMPLATE.md
├── PRESENTATION_OUTLINE.md
└── requirements.txt
```

---

## 18. Model Files

The trained model artifacts are stored in the `models` directory.

### `mlp.pt`

Contains the trained PyTorch neural-network model.

### `pca.joblib`

Contains the trained PCA transformation.

It is used to transform the clothing representation into PCA space and reconstruct the predicted representation using inverse PCA.

### `x_scaler.joblib`

Contains the trained input scaler.

It is used to transform the 56-dimensional pose input before neural-network inference.

---

## 19. Results Files

The `results` directory contains the generated experimental outputs.

### `loss_curve.png`

Training and validation loss visualization.

### `loss_history.csv`

Training and validation loss values recorded during training.

### `metrics.txt`

Evaluation metrics generated by the experiment.

### `pca_variance.png`

Cumulative PCA explained-variance visualization.

---

## 20. Dataset

The current implementation uses a surrogate dataset for the machine-learning experiment.

The surrogate dataset provides the required input-output structure for developing and testing the complete pipeline.

The current experiment contains:

```text
3,000 total samples

1,800 training samples
600 validation samples
600 test samples
```

The input representation contains:

```text
56 features
```

and the original output representation contains:

```text
6,510 values
```

---

## 21. Dataset Scope

The current implementation should be considered a machine-learning research prototype.

The current results are based on the available surrogate dataset and therefore should not be interpreted as results obtained from a real-world physics-based clothing simulation dataset.

A future version can replace the surrogate data with an actual physics-simulated clothing dataset while keeping the same general machine-learning pipeline.

---

## 22. Limitations

The current implementation has several limitations.

### 1. Surrogate Dataset

The current experiment uses a surrogate dataset rather than the original physics-simulated clothing dataset.

### 2. Limited Physical Interpretation

The predicted 6,510-dimensional representation is treated as a numerical clothing-deformation representation.

The current implementation does not perform a complete physical cloth simulation.

### 3. Dataset Size

The current experiment uses 3,000 samples.

A larger dataset could be used to investigate model generalization further.

### 4. Model Comparison

The current experiment compares the neural network with a linear-regression baseline.

Additional machine-learning and deep-learning models could be evaluated in future work.

---

## 23. Future Scope

The project can be extended in several ways.

### Real Physics-Simulation Dataset

The surrogate dataset can be replaced with a real physics-simulated clothing dataset.

### Larger Dataset

A larger and more diverse dataset can be used to improve the robustness of the learned mapping.

### Advanced Neural Networks

Future experiments can investigate:

- Deeper fully connected networks
- Residual networks
- Autoencoders
- Convolutional architectures
- Transformer-based models

### Additional Baselines

Additional regression and machine-learning algorithms can be evaluated against the neural network.

### Improved Clothing Visualization

The predicted 6,510-dimensional representation can be connected to a 3D clothing mesh to visualize the actual deformation.

### Real-Time Inference

The trained model can potentially be optimized for real-time prediction from pose data.

### Physics-Based Validation

Predicted clothing deformation can be compared directly against physics-simulation ground truth using additional geometric metrics.

---

## 24. Technologies Used

The project uses the following technologies:

- Python
- NumPy
- Pandas
- PyTorch
- Scikit-learn
- Joblib
- Streamlit
- Plotly
- Matplotlib

---

## 25. Installation

Create and activate a Python virtual environment if required.

Then install the project dependencies:

```bash
pip install -r requirements.txt
```

---

## 26. Running the Application

Navigate to the project directory:

```bash
cd data_driven_clothing_project
```

Then run:

```bash
streamlit run app.py
```

The Streamlit server will start and provide a local URL.

Open the displayed URL in a browser to access the ClothAI dashboard.

---

## 27. Running the Training Pipeline

The project training pipeline can be executed through the application according to the current implementation.

During training, the terminal displays epoch-wise training and validation loss.

Example:

```text
Epoch 080 | train=0.058521 | val=0.079333
Epoch 081 | train=0.059636 | val=0.079866
Epoch 082 | train=0.058882 | val=0.077908
...
Early stopping.
Training complete.
```

The trained artifacts and result files are generated inside the project directories.

---

## 28. Reproducibility

The project stores the important preprocessing and model components as files:

```text
models/mlp.pt
models/pca.joblib
models/x_scaler.joblib
```

This allows the Streamlit application to use the trained model and preprocessing pipeline during inference.

The project also stores the generated plots and evaluation results.

---

## 29. End-to-End Workflow

The complete project workflow is:

```text
Dataset
   ↓
Data Preparation
   ↓
Train / Validation / Test Split
   ↓
Input Standardization
   ↓
PCA on Clothing Representation
   ↓
128 PCA Components
   ↓
Neural Network Training
   ↓
Early Stopping
   ↓
Model Evaluation
   ↓
Save Model Artifacts
   ↓
Streamlit Dashboard
   ↓
User Provides 56-Dimensional Pose
   ↓
Input Scaling
   ↓
Neural Network Prediction
   ↓
Predicted PCA Coefficients
   ↓
Inverse PCA
   ↓
6,510-Dimensional Prediction
   ↓
Visualization and Summary
```

---

## 30. Key Experimental Values

```text
Input features                  : 56
Original output dimensions      : 6,510
PCA components                  : 128
Training samples                : 1,800
Validation samples              : 600
Testing samples                 : 600
Total samples                   : 3,000

PCA explained variance          : 0.9998849

Neural Network MSE              : 0.00150401
Neural Network RMSE             : 0.03878156
Neural Network MAE              : 0.03073971

Linear Regression MSE           : 0.00109090
```

---

## 31. Conclusion

ClothAI demonstrates an end-to-end machine-learning pipeline for learning a mapping between human-pose features and a high-dimensional clothing representation.

The project combines:

```text
Input preprocessing
        +
PCA dimensionality reduction
        +
Neural-network learning
        +
Inverse PCA reconstruction
        +
Model evaluation
        +
Interactive Streamlit inference
```

The current experiment successfully trains the pipeline, stores the trained model artifacts, generates evaluation results, and provides an interactive dashboard for prediction and visualization.

The project serves as a research prototype for data-driven clothing deformation prediction and provides a foundation for future experiments using larger and physically simulated clothing datasets.

---

## 32. Project Status

**Completed Research Prototype**

Current implementation includes:

- Data preparation
- Train/validation/test split
- Input scaling
- PCA dimensionality reduction
- Neural-network training
- Early stopping
- Linear-regression baseline
- Model evaluation
- Model artifact saving
- Training-loss visualization
- PCA explained-variance visualization
- Streamlit dashboard
- Live prediction
- Prediction visualization
- Prediction summary
- CSV prediction export
- Methodology documentation
- Dataset and scope documentation

---

## 33. Quick Start

For a quick start:

```bash
cd data_driven_clothing_project
pip install -r requirements.txt
streamlit run app.py
```

Then open the Streamlit URL shown in the terminal.

---

## 34. Final Project Pipeline

```text
                 CLOTHAI
                    │
                    ▼
          56-D Human Pose Input
                    │
                    ▼
            Input Standardization
                    │
                    ▼
              PCA Processing
                    │
                    ▼
             128 PCA Components
                    │
                    ▼
          Fully Connected MLP
                    │
                    ▼
        Predicted PCA Coefficients
                    │
                    ▼
              Inverse PCA
                    │
                    ▼
        6,510-D Clothing Output
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
      Evaluation          Visualization
          │                   │
          ▼                   ▼
       Metrics          Streamlit Dashboard
```

---

**ClothAI — Data-Driven Clothing Prediction**

**Machine Learning Mini-Project • Research Prototype**
