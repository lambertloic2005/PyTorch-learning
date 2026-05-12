import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

x = torch.tensor([1.0, 2.0, 3.0])

print("x:", x)
print("x shape:", x.shape)
print("x mean:", x.mean())