import torch
import decord

pictures = torch.randint(0, 256, (1000, 28, 28, 3))
print(pictures)
print("hello world")