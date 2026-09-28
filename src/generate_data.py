import os
import numpy as np

SEED = 42
N = 3000
INPUT_DIM = 56
OUTPUT_DIM = 6510
LATENT_DIM = 128

rng = np.random.default_rng(SEED)
os.makedirs("data", exist_ok=True)

# Synthetic surrogate: generate pose vectors that look like normalized joint features.
X = rng.normal(0, 1, size=(N, INPUT_DIM)).astype(np.float32)
X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-6)

# Low-dimensional nonlinear garment deformation basis.
B = rng.normal(0, 0.03, size=(LATENT_DIM, OUTPUT_DIM)).astype(np.float32)
W1 = rng.normal(0, 0.7, size=(INPUT_DIM, LATENT_DIM)).astype(np.float32)
W2 = rng.normal(0, 0.25, size=(INPUT_DIM, LATENT_DIM)).astype(np.float32)

Z = np.tanh(X @ W1) + 0.15 * np.sin(X @ W2)
Y = (Z @ B + 0.002 * rng.normal(size=(N, OUTPUT_DIM))).astype(np.float32)

np.save("data/X.npy", X)
np.save("data/Y.npy", Y)
print(f"Generated synthetic surrogate dataset: X={X.shape}, Y={Y.shape}")
print("NOTE: This is not the original paper dataset; it is for demonstrating the assigned ML pipeline.")
