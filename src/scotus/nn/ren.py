from typing import Callable

import torch
import torch.nn.functional as F


class DynamicMemory(torch.nn.Module):
    """
    """
    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        d_key: int,
        act_hidden: Callable[..., torch.Tensor],
        act_out: Callable[..., torch.Tensor],
    ):
        # Hidden state learning parameters
        self.linear_hidden = torch.nn.Linear(d_hidden, d_hidden, bias=False)
        self.linear_key = torch.nn.Linear(d_key, d_hidden, bias=False)
        self.linear_input = torch.nn.Linear(d_in, d_hidden, bias=False)

        # States don't require_grad because we manually update them in the forward
        # pass in training.
        self.G_gates, self.K_keys = (
            torch.nn.Parameter(torch.zeros((d_in, d_hidden)), requires_grad=False),
            torch.nn.Parameter(torch.ones((d_key, d_hidden)), requires_grad=False),
        )

        # 
        self.act_hidden = act_hidden
        self.act_out = act_out

    def forward(self, x_t, H_t):
        self.G_gates = torch.sigmoid(x_t.T @ H_t + x_t.T @ self.K_keys)
        H_candidate = self.act_hidden(
            self.linear_hidden(self.H_hidden)
            + self.linear_key(self.K_keys)
            + self.linear_input(x_t)
        )
        return F.normalize(H_t + self.G_gates * H_candidate)

