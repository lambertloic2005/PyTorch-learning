import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# -------------------------
# 1. Make fake classification data
# -------------------------

num_points = 500

class_0 = torch.randn(num_points, 2) + torch.tensor([-2.0, -2.0])
class_1 = torch.randn(num_points, 2) + torch.tensor([2.0, 2.0])

x = torch.cat([class_0, class_1], dim=0)

y0 = torch.zeros(num_points, dtype=torch.long)
y1 = torch.ones(num_points, dtype=torch.long)
y = torch.cat([y0, y1], dim=0)

print("x shape:", x.shape)
print("y shape:", y.shape)

dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# -------------------------
# 2. Model
# -------------------------

model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 2),
)

# Output shape is [batch_size, 2]
# The 2 outputs are raw class scores, also called logits.

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# -------------------------
# 3. Training
# -------------------------

for epoch in range(200):
    total_loss = 0.0
    correct = 0
    total = 0

    for x_batch, y_batch in dataloader:
        logits = model(x_batch)
        loss = loss_fn(logits, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == y_batch).sum().item()
        total += y_batch.shape[0]

    accuracy = correct / total

    if epoch % 20 == 0:
        print(
            f"Epoch {epoch}, "
            f"loss = {total_loss / len(dataloader):.6f}, "
            f"accuracy = {accuracy:.3f}"
        )

# -------------------------
# 4. Test individual points
# -------------------------

model.eval()

with torch.no_grad():
    test_points = torch.tensor([
        [-3.0, -3.0],
        [3.0, 3.0],
        [-2.0, 1.0],
        [2.0, -1.0],
    ])

    logits = model(test_points)
    predictions = torch.argmax(logits, dim=1)

print("\nTest points:")
print(test_points)

print("\nPredicted classes:")
print(predictions)