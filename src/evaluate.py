import os
import numpy as np
import joblib
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from train import MLP

os.makedirs("results", exist_ok=True)

SEED = 42
X = np.load("data/X.npy")
Y = np.load("data/Y.npy")

# Exactly reproduce train.py split.
X_train, X_tmp, Y_train, Y_tmp = train_test_split(X, Y, test_size=0.40, random_state=SEED)
X_val, X_test, Y_val, Y_test = train_test_split(X_tmp, Y_tmp, test_size=0.50, random_state=SEED)

x_scaler = joblib.load("models/x_scaler.joblib")
pca = joblib.load("models/pca.joblib")

X_test_s = x_scaler.transform(X_test)
Z_train = pca.transform(Y_train)
Z_test = pca.transform(Y_test)

# Neural network
model = MLP()
model.load_state_dict(torch.load("models/mlp.pt", map_location="cpu"))
model.eval()
with torch.no_grad():
    pred_z = model(torch.tensor(X_test_s, dtype=torch.float32)).numpy()
pred_y = pca.inverse_transform(pred_z)

mse = mean_squared_error(Y_test, pred_y)
rmse = np.sqrt(mse)
mae = mean_absolute_error(Y_test, pred_y)

# Linear regression baseline in PCA space.
lr = LinearRegression()
lr.fit(x_scaler.transform(X_train), Z_train)
lr_pred_y = pca.inverse_transform(lr.predict(X_test_s))
lr_mse = mean_squared_error(Y_test, lr_pred_y)

print(f"Neural Network Test MSE:  {mse:.8f}")
print(f"Neural Network Test RMSE: {rmse:.8f}")
print(f"Neural Network Test MAE:  {mae:.8f}")
print(f"Linear Regression Test MSE: {lr_mse:.8f}")

with open("results/metrics.txt", "w") as f:
    f.write(f"Neural Network Test MSE: {mse:.8f}\n")
    f.write(f"Neural Network Test RMSE: {rmse:.8f}\n")
    f.write(f"Neural Network Test MAE: {mae:.8f}\n")
    f.write(f"Linear Regression Test MSE: {lr_mse:.8f}\n")
    f.write(f"PCA explained variance (128 components): {pca.explained_variance_ratio_.sum():.6f}\n")

hist = np.loadtxt("results/loss_history.csv", delimiter=",", skiprows=1)
plt.figure()
plt.plot(hist[:,0], hist[:,1], label="Train Loss")
plt.plot(hist[:,0], hist[:,2], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig("results/loss_curve.png", dpi=180)
plt.close()

plt.figure()
plt.plot(np.arange(1, len(pca.explained_variance_ratio_)+1),
         np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Number of PCA components")
plt.ylabel("Cumulative explained variance")
plt.title("PCA Explained Variance")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/pca_variance.png", dpi=180)
plt.close()
