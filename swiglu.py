import math
import torch
import torch.nn as nn

class Swish(nn.Module):
    """
    This class implements the swish function, which
    is defined as swish(x) = x * sigmoid(x).

    Sources:
    
    https://arxiv.org/pdf/1710.05941v1
    cite par
    https://openreview.net/pdf?id=OvoCm1gGhN
    """
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return self.sigmoid(x) * x

class SwiGLU(nn.Module):
    """
    This class implements the SwiGLU function using more basic modules.
    This function is defined in the main paper we are using and elsewhere.

    Source:

    https://openreview.net/pdf?id=OvoCm1gGhN line 181

    Code style, structure and choices taken from nn.Linear implementation
    """

    __constants__ = ["d_model"]
    d_model: int
    w_g: torch.Tensor
    w_1: torch.Tensor
    w_2: torch.Tensor


    def __init__(
        self,
        d_model: int,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.w_g = nn.Parameter(
            torch.empty((d_model, d_model * 8 // 3), **factory_kwargs)
        )
        self.w_1 = nn.Parameter(
            torch.empty((d_model, d_model * 8 // 3), **factory_kwargs)
        )
        self.w_2 = nn.Parameter(
            torch.empty((d_model * 8 // 3, d_model), **factory_kwargs)
        )
        self.swish = Swish()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Critere similaire au critere d'initialisation des reseaux lineaires
        nn.init.kaiming_uniform_(self.w_g, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w_1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w_2, a=math.sqrt(5))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.swish(x @ self.w_g) * (x @ self.w_1)) @ self.w_2