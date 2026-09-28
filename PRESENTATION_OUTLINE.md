# Presentation Outline — Data-Driven Clothing for Interactive Applications

## Slide 1 — Title
Data-Driven Clothing for Interactive Applications
Machine Learning Mini-Project

## Slide 2 — Problem Statement
- Physics-based cloth simulation can be computationally expensive for interactive applications.
- Goal: learn a data-driven mapping from human pose to clothing deformation.

## Slide 3 — Reference Problem Structure
- 56 pose features.
- 2,170 clothing vertices × 3 coordinates = 6,510 output values.
- Reference paper uses PCA + fully connected neural network.

## Slide 4 — Dataset and Scope
- Raw reference dataset not supplied with assignment.
- Synthetic surrogate: 3,000 samples.
- Train/validation/test = 1,800/600/600.
- Same 56 → 6,510 dimensional structure.

## Slide 5 — Proposed Pipeline
56-D pose → Standardization → PCA 6,510→128 → MLP 56→128→256→128 → inverse PCA → 6,510-D prediction

## Slide 6 — Model and Training
- PyTorch MLP
- ReLU activations
- MSE loss
- Adam optimizer
- Batch size 32
- Early stopping

## Slide 7 — Results
- PCA explained variance: 99.9885%
- NN MSE: 0.00150401
- NN RMSE: 0.03878156
- NN MAE: 0.03073971
- Linear regression MSE: 0.00109090
- Include loss and PCA plots.

## Slide 8 — Live Demonstration
Show the Streamlit dashboard: Live Prediction, Results, Methodology and Dataset tabs.

## Slide 9 — Limitations and Future Work
- Surrogate data rather than original physics-simulated dataset.
- No original mesh topology supplied, so current UI reports the 6,510-D representation rather than claiming a true garment render.
- Future: original dataset, true 3D mesh visualization, temporal sequences, architecture/PCA ablations.

## Slide 10 — Conclusion
The project demonstrates a reproducible PCA + neural-network pipeline for fast prediction of high-dimensional clothing deformation from pose features.
