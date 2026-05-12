import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# -------------------------
# 1. Data
# -------------------------

x = torch.linspace(-5, 5, 200).reshape(-1, 1)
y = x ** 2

dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# -------------------------
# 2. Model
# -------------------------

model = nn.Sequential(
    nn.Linear(1, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# -------------------------
# 3. Training with batches
# -------------------------

for epoch in range(500):
    total_loss = 0.0

    for x_batch, y_batch in dataloader:
        prediction = model(x_batch)
        loss = loss_fn(prediction, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(dataloader)

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, average loss = {average_loss:.6f}")

# -------------------------
# 4. Test
# -------------------------

model.eval()

with torch.no_grad():
    test_x = torch.tensor([[2.0]])
    test_prediction = model(test_x)

print("\nPrediction for x = 2:", test_prediction.item())
print("Correct answer:", 4.0)