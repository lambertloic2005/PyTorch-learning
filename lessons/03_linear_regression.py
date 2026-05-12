import torch
import torch.nn as nn

# -------------------------
# 1. Training data
# -------------------------

x = torch.tensor([
    [1.0],
    [2.0],
    [3.0],
    [4.0],
])

y = torch.tensor([
    [3.0],
    [5.0],
    [7.0],
    [9.0],
])

# The hidden rule is:
# y = 2x + 1

# -------------------------
# 2. Model
# -------------------------

model = nn.Linear(1, 1)

# This model represents:
# prediction = weight * x + bias

# -------------------------
# 3. Loss function
# -------------------------

loss_fn = nn.MSELoss()

# MSE = mean squared error
# It punishes predictions that are far away from the true answer.

# -------------------------
# 4. Optimizer
# -------------------------

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# SGD = stochastic gradient descent
# lr = learning rate

# -------------------------
# 5. Training loop
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
# 6. Inspect result
# -------------------------

print("\nLearned weight:", model.weight.item())
print("Learned bias:", model.bias.item())

test_x = torch.tensor([[10.0]])
test_prediction = model(test_x)

print("\nPrediction for x = 10:", test_prediction.item())
print("Correct answer:", 21.0)