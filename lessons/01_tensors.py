import torch

# A scalar: one number
scalar = torch.tensor(3.0)

# A vector: one-dimensional list of numbers
vector = torch.tensor([1.0, 2.0, 3.0])

# A matrix: two-dimensional table of numbers
matrix = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0],
])

print("scalar:", scalar)
print("vector:", vector)
print("matrix:")
print(matrix)

print("\nShapes:")
print("scalar shape:", scalar.shape)
print("vector shape:", vector.shape)
print("matrix shape:", matrix.shape)

print("\nBasic operations:")
print("vector + vector:", vector + vector)
print("vector * 10:", vector * 10)
print("matrix mean:", matrix.mean())
print("matrix sum:", matrix.sum())

print("\nMatrix multiplication:")
A = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0],
])

B = torch.tensor([
    [5.0],
    [6.0],
])

print(A @ B)