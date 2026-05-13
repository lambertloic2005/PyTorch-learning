import torch
import torch.nn as nn
from pathlib import Path

# -------------------------
# 1. Create output folder
# -------------------------

Path("outputs").mkdir(exist_ok=True)

# -------------------------
# 2. Data
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

# -------------------------
# 3. Train model
# -------------------------

model = nn.Linear(1, 1)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    prediction = model(x)
    loss = loss_fn(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# -------------------------
# 4. Save model parameters
# -------------------------

save_path = "outputs/linear_model.pt"
torch.save(model.state_dict(), save_path)

print("Saved model to:", save_path)

# -------------------------
# 5. Load model parameters into a new model
# -------------------------

new_model = nn.Linear(1, 1)
new_model.load_state_dict(torch.load(save_path))

new_model.eval()

with torch.no_grad():
    test_x = torch.tensor([[10.0]])
    prediction = new_model(test_x)

print("Loaded model prediction for x = 10:", prediction.item())