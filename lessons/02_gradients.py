import torch

# requires_grad=True tells PyTorch to track operations involving x
x = torch.tensor(2.0, requires_grad=True)

# Define a simple function
y = 3*x**2+2*x+1

# Compute dy/dx
y.backward()

print("x:", x)
print("y:", y)
print("dy/dx:", x.grad)