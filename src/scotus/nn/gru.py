from typing import Callable

import torch
import torch.nn.functional as F


class Gate(torch.nn.Module):
    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        act_gate: Callable[..., torch.Tensor] = F.sigmoid,
    ):
        self.d_in, self.d_hidden = d_in, d_hidden
        self.W, self.U = (
            torch.nn.Linear(d_in, d_hidden),
            torch.nn.Linear(d_hidden, d_hidden),
        )
        self.activation = act_gate
        self.init_weights()

    def init_weights(self):
        k = 1 / self.d_hidden
        for param in self.parameters(recurse=True):
            torch.nn.init.uniform_(param, a=-torch.sqrt(k), b=torch.sqrt(k))

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self.act(self.W(x) + self.U(h))


class GRU(torch.nn.Module):
    """
    A GatedRecurrentUnit (GRU) uses two Gate modules, activated with
    sigmoids, to maintain relevance and decide which information from
    history is important to making a decision in the current time step.

    Reference:
        https://towardsdatascience.com/understanding-gru-networks-2ef37df6c9be

    """
    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        d_out: int,
        act: Callable = F.tanh,
        act_update: Callable = F.sigmoid,
        act_forget: Callable = F.sigmoid,
    ):
        # Construct gates
        self.update_gate = Gate(d_in, d_hidden, act_gate=act_update)
        self.forget_gate = Gate(d_in, d_hidden, act_gate=act_forget)
        self.activation = act

        self.update, self.forget = (
            torch.nn.Parameter(torch.empty(d_out, ), requires_grad=False),
            torch.nn.Parameter(torch.empty(d_out, ), requires_grad=False),
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # Roll the gated vectors forward a time step so we can update history
        self.update = self.update_gate(x, h)
        self.forget = self.forget_gate(x, h)

        # Roll the history forward using the reset gate, and trade off between
        # the update gate
        h = self.act((self.W @ x) + (self.forget * (self.U @ h)))
        h = ((self.update * h) + ((1 - self.update) * h))
        z = self.act((self.W @ x) + (self.U @ h) + self.b)
        return z, h
