import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy

# -------------------------
# 1. Data
# -------------------------

x = torch.linspace(-5, 5, 200).reshape(-1, 1)
y = x ** 2

print("x shape:", x.shape)
print("y shape:", y.shape)

# -------------------------
# 2. Neural network
# -------------------------

model = nn.Sequential(
    nn.Linear(1, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
)

print(model)

# -------------------------
# 3. Loss and optimizer
# -------------------------

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# -------------------------
# 4. Training
# -------------------------

for epoch in range(1000):
    prediction = model(x)
    loss = loss_fn(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, loss = {loss.item():.6f}")

# -------------------------
# 5. Plot result
# -------------------------

model.eval()

with torch.no_grad():
    prediction = model(x)

plt.scatter(x.numpy(), y.numpy(), s=10, label="True data")
plt.plot(x.numpy(), prediction.numpy(), label="Model prediction")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.savefig("outputs/04_neural_network_result.png")
plt.show()