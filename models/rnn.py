"""

https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks
"""
import torch
import torch.nn.functional as F
from torch import nn


class RNN(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        d_out: int,
        act_hidden: Callable = F.sigmoid,
        act_out: Callable = F.sigmoid,
    ):
        self.act_hiddem = act_hidden
        self.act_out = act_out

        # Learned weight parameters
        self.W_ax, self.W_aa, self.W_ya = (
            nn.Parameter(torch.zeros(d_hidden, d_in)),
            nn.Parameter(torch.ones(d_hidden, d_hidden)),
            nn.Parameter(torch.normal(d_out, d_hidden)),
        )

        # Static bias parameters
        self.b_a, self.b_y = (
            nn.Parameter(torch.ones(d_hidden), requires_grad=False),
            nn.Parameter(torch.ones(d_hidden), requires_grad=False),
        )

        # Stored context tensor
        self.hidden = nn.Parameter(torch.zeros(d_hidden), requires_grad=False)

    def forward(self, x: torch.tensor):
        # Update the attention/contex hidden vector with the new word
        self.hidden = self.act_hidden(
            (self.W_aa @ self.hidden)
            + (self.W_ax @ x)
            + self.b_a
        )
        y = self.act_out(self.W_ya @ self.hidden + self.b_y)
        return y


if __name__ == "__main__":
    rnn = RNN(50, 20, 50, nn.ReLU(), nn.ReLU())
