"""
https://towardsdatascience.com/understanding-gru-networks-2ef37df6c9be
"""
import torch
import torch.nn.functional as F
from torch import nn

from . import RNN


class Gate(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, act: Callable = F.sigmoid):
        self.W, self.U, self.b = (
            nn.Parameter(torch.uniform((d_in, d_hidden))),
            nn.Parameter(torch.uniform((d_hidden, d_hidden))),
            nn.Parameter(torch.ones(d_hidden, requires_grad=False))
        )
        self.act = act

    def forward(self, x: torch.tensor, h: torch.tensor) -> torch.tensor:
        return self.act((self.W @ x) + (self.U @ h) + self.b)


class GRU(RNN):
    """
    A GatedRecurrentUnit (GRU) uses two Gate modules, activated with
    sigmoids, to maintain relevance and decide which information from
    history is important to making a decision in the current time step.
    """
    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        d_out: int,
        act: Callable = F.tanh,
        act_upd: Callable = F.sigmoid,
        act_rel: Callable = F.sigmoid,
    ):
        # Initialize standard recurrent parameters
        super().__init__(d_in, d_hidden, d_out, act)

        # Construct gates
        self.update_gate = Gate(d_in, d_hidden, act=act_upd)
        self.reset_gate = Gate(d_in, d_hidden, act=act_rel)

        self.update, self.reset = (
            nn.Parameter(torch.full((d_out,), torch.nan), requires_grad=False),
            nn.Parameter(torch.full((d_out,), torch.nan), requires_grad=False),
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        # Roll the gated vectors forward a time step so we can update history
        self.update = self.update_gate(x, self.hidden)
        self.reset = self.reset_gate(x, self.hidden)

        # Roll the history forward using the reset gate, and trade off between
        # the update gate
        hidden = self.act((self.W @ x) + (self.reset * (self.U @ self.hidden)))
        self.hidden = (
            (self.update * hidden) + ((1 - self.update) * self.hidden)
        )
        y = super().forward(x)
        return y