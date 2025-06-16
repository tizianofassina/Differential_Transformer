import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """Root Mean Square Normalization layer.

    https://arxiv.org/abs/1910.07467
    """
    def __init__(self, layer_dim, eps: float=1e-7):
        super().__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(layer_dim))
    
    def norm(self, x):
        return torch.sqrt(torch.mean(x.square(), dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        return (x.float() / self.norm(x) * self.gain).type_as(x)

class GroupNorm(nn.Module):
    """Group Nomalization layer.

    https://arxiv.org/abs/1803.08494

    The design was adjusted to follow the specifications of
    https://arxiv.org/abs/2410.05258.
    """
    def __init__(self, head_dim, lambda_init, eps=1e-7):
        super().__init__()

        self.lambda_init = lambda_init
        self.norm = RMSNorm(head_dim, eps)

    def forward(self, x):
        return (1 - self.lambda_init) * self.norm.forward(x)