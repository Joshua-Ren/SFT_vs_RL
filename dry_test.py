import torch
from utils.hh_metrics import (
    compute_pairwise_cosine,
    compute_pairwise_dot,
    summarize_alignment_values,
)

H = torch.tensor([
    [1.0, 0.0],   # 0
    [0.0, 1.0],   # 1
    [-1.0, 0.0],  # 2
    [1.0, 1.0],   # 3
])

pairs = torch.tensor([
    [0, 0],  # same vector
    [0, 1],  # orthogonal
    [0, 2],  # opposite
    [0, 3],  # 45 degree
])

cos = compute_pairwise_cosine(H, pairs, centered=False)
dot = compute_pairwise_dot(H, pairs, centered=False)

print("cos:", cos)
print("dot:", dot)
print("summary:", summarize_alignment_values(cos))