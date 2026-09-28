# Data-Driven Clothing for Interactive Applications

## 1. Problem Statement
Physics-based cloth simulation can be computationally expensive for interactive applications. The assigned research paper investigates a data-driven alternative that learns to predict clothing deformation from human-pose parameters. This mini-project implements the central machine-learning pipeline: pose representation → PCA-based clothing compression → fully connected neural network → reconstructed clothing-offset vector.

## 2. Dataset and Scope
The reference paper describes 10,000 simulated coat examples, with 56 pose inputs and 6,510 clothing-offset outputs (2,170 vertices × x,y,z), with 9,813 examples retained after filtering. The raw physics-simulated dataset was not included with the supplied assignment material. Therefore, this implementation uses a synthetic surrogate dataset that preserves the same dimensional structure. The surrogate contains 3,000 samples split into 1,800 training, 600 validation and 600 test samples.

## 3. Methodology
The 56-dimensional pose input is standardized using training-set statistics. PCA is fitted only on the training clothing outputs to avoid data leakage and compresses the 6,510-dimensional representation into 128 components. A PyTorch multilayer perceptron predicts the 128 PCA coefficients from the 56 pose features. The network contains fully connected layers of 128 and 256 neurons with ReLU activations, followed by the 128-dimensional output layer. Training uses MSE loss, the Adam optimizer, batch size 32 and early stopping. The predicted PCA coefficients are transformed back into the 6,510-dimensional clothing representation using inverse PCA.

## 4. Implementation and Demonstration
The implementation uses Python, NumPy, scikit-learn, PyTorch, Matplotlib, Plotly and Streamlit. The repository contains separate scripts for surrogate-data generation, training, evaluation and a Streamlit demonstration. The UI provides pose presets/custom input, live prediction, output statistics, CSV download, loss/PCA visualizations, baseline comparison and methodology/dataset documentation.

## 5. Results
The 128-component PCA retained 99.9885% of the variance in the training clothing data. On the held-out test set, the neural network achieved MSE = 0.00150401, RMSE = 0.03878156 and MAE = 0.03073971. The linear-regression baseline achieved MSE = 0.00109090. Thus, the neural network did not obtain a lower MSE than the linear baseline in this particular surrogate-data experiment. This result is retained as an experimental finding rather than being replaced by a claim of superiority.

## 6. Conclusion and Future Work
The project demonstrates an end-to-end data-driven clothing prediction workflow and shows how PCA can reduce a high-dimensional clothing representation before neural-network prediction. The current implementation is a faithful demonstration of the ML pipeline at the dimensional level, but it is not a full reproduction of the paper because the original physics-simulated dataset and garment mesh topology were not supplied. Future work should use the original dataset, perform true 3D garment reconstruction, compare several PCA/model configurations, and evaluate temporal pose sequences for interactive applications.
