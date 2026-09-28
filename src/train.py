import os
import numpy as np
import joblib
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

X = np.load("data/X.npy")
Y = np.load("data/Y.npy")

X_train, X_tmp, Y_train, Y_tmp = train_test_split(X, Y, test_size=0.40, random_state=SEED)
X_val, X_test, Y_val, Y_test = train_test_split(X_tmp, Y_tmp, test_size=0.50, random_state=SEED)

x_scaler = StandardScaler()
X_train_s = x_scaler.fit_transform(X_train)
X_val_s = x_scaler.transform(X_val)
X_test_s = x_scaler.transform(X_test)

# PCA is fitted ONLY on training clothing outputs to avoid leakage.
pca = PCA(n_components=128, svd_solver="randomized", random_state=SEED)
Z_train = pca.fit_transform(Y_train)
Z_val = pca.transform(Y_val)
Z_test = pca.transform(Y_test)

joblib.dump(x_scaler, "models/x_scaler.joblib")
joblib.dump(pca, "models/pca.joblib")

class MLP(nn.Module):
    def __init__(self, in_dim=56, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )
    def forward(self, x):
        return self.net(x)

model = MLP()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

Xt = torch.tensor(X_train_s, dtype=torch.float32)
Zt = torch.tensor(Z_train, dtype=torch.float32)
Xv = torch.tensor(X_val_s, dtype=torch.float32)
Zv = torch.tensor(Z_val, dtype=torch.float32)

loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(Xt, Zt), batch_size=32, shuffle=True
)

best = float("inf")
patience = 15
wait = 0
history = []

for epoch in range(1, 101):
    model.train()
    total = 0.0
    for xb, yb in loader:
        opt.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        opt.step()
        total += loss.item() * len(xb)
    train_loss = total / len(Xt)

    model.eval()
    with torch.no_grad():
        val_loss = loss_fn(model(Xv), Zv).item()

    history.append((epoch, train_loss, val_loss))
    print(f"Epoch {epoch:03d} | train={train_loss:.6f} | val={val_loss:.6f}")

    if val_loss < best:
        best = val_loss
        wait = 0
        torch.save(model.state_dict(), "models/mlp.pt")
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping.")
            break

np.savetxt("results/loss_history.csv", np.array(history), delimiter=",",
           header="epoch,train_loss,val_loss", comments="")

print("Training complete.")
print("Train/Val/Test:", len(X_train), len(X_val), len(X_test))
print("PCA explained variance:", pca.explained_variance_ratio_.sum())
