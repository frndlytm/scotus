from typing import Callable

import torch
import torch.nn.functional as F

from .rand import uniform


class RNNCell(torch.nn.Modeule):
    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        act_hidden: Callable[..., torch.Tensor] = F.identity,
        bias: bool = False
    ):
        self.activation = act_hidden
        self.W_hidden = torch.nn.Linear(d_hidden, d_hidden, bias=bias)
        self.W_input = torch.nn.Linear(d_in, d_hidden, bias=bias)

    def forward(self, hidden: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Update the attention/contex hidden vector with the new word
        return self.activation(self.W_hidden(hidden) + self.W_input(x))


# Vanilla RNN Implementation
class RNN(torch.nn.Module):
    """
    Reference:
        https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks
    """
    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        d_out: int,
        act_hidden: Callable[..., torch.Tensor] = F.sigmoid,
        act_out: Callable[..., torch.Tensor] = F.sigmoid,
        bias: bool = False,
    ):
        self.hidden, self.hidden_update = (
            torch.nn.Parameter(uniform, ),
            RNNCell(d_in, d_hidden, act_hidden, bias=bias),
        )

        self.out = torch.nn.Linear(d_hidden, d_out, bias=bias)
        self.activation = act_out

    def forward(self, x: torch.tensor):
        self.hidden = self.act_hidden(
            (self.W_aa @ self.hidden)
            + (self.W_ax @ x)
            + self.b_a
        )
        y = self.act_out(self.W_ya @ self.hidden + self.b_y)
        return y
