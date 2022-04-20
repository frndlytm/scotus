
from typing import Callable

import torch
import torch.nn.funtional as F
from torch import nn



# TODO:
#     TITLE: CNN For Sentence Embedding for Utterances
#     AUTHOR: frndlytm
class MaxOverTimePooling(nn.Module):
    def forward(self, x):
        return nn.max


class CNNSentenceClassifier(nn.Module):
    def __init__(self, k_width: int, *h_windows: int):
        super().__init__()
        self.k_width = k_width
        self.embeddings = nn.Embedding(...)
        self.convolution = nn.ModuleList([
            nn.Conv1D(h_window) for h_window in h_windows
        ])
        self.max_over_time = nn.MaxPool1d(...)

    def forward(self, utterance):
        """
        Given an input sentence, x, a CNNSentenceClassifier convolves the
        word vectors into channels by context h_windows.
        """
        embedded = self.embedding(utterance)
        xs = self.convolution(embedded)


# Vanilla RNN Implementation
class RNN(nn.Module):
    """
    Reference:
        https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks
    """
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

    Reference:
        https://towardsdatascience.com/understanding-gru-networks-2ef37df6c9be

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


# TODO:
#     TITLE: SpeakerUpdate GRU from Paper
#     AUTHOR: frndlytm
class SpeakerSubmodule(GRU):
    def __init__(
        self,
        name: str,
        d_global: int,
        d_utter: int,
        d_history: int,
    ):
        self.name = name
        self.emotion = GRU(...)
        self.history = GRU(...)

        self.alpha = nn.Parameter(torch.ones(d_history), requires_grad=False)
        self.context = nn.Parameter(torch.zeros(d_global), requires_grad=False)
        self.W_alpha = nn.Parameter(torch.normal((d_utter, d_global)))
        super().__init__(...)

    def forward(
        self,
        utter: torch.tensor,
        previous_state: torch.tensor,
        global_history: torch.tensor
    ):
        self.alpha = F.softmax(utter @ self.W_alpha @ global_history)
        self.context = self.alpha @ global_history.T
        return super().forward(
            previous_state, torch.concat(utter, self.context)
        )


class DialgueRNN(nn.Module):
    def __init__(
        self,
        parties: set[str],
        d_global: int,
        d_party: int,
        d_utter: int,
        d_history: int,
    ):
        # Track the state of who's talking and who's listening
        self.channels = len(parties)
        self.is_speaker = torch.zeros(self.channels)

        # Configure parties' individual hidden contexts
        self.party = {p: i for i, p in enumerate(parties)}
        self.party_state = nn.Parameter(torch.zeros(self.channels, d_party))
        self.party_grus = torch.ModuleList([
            SpeakerSubmodule(name, d_global, d_utter, d_history)
            for name in parties
        ])

        # Configure global conversation state module and an empty history
        # of size d_history
        self.global_gru = GRU(d_global, d_history)
        self.global_history = torch.zeros((d_history, d_global))

    @property
    def is_listener(self):
        return 1 - self.is_speaker

    def set_speaker(self, speaker):
        # reset the speakers tensor to 0's
        self.is_speakers = torch.zeros_like(self.is_speakers)

        # set the current active speaker using the participants index
        i = self.participants[speaker]
        self.is_speakers[i] = 1

    def forward(self, speaker: str, utterance: torch.tensor):
        self.set_speaker(speaker)
        # TODO:
        #     TITLE: DialogueRNN forward
        #     AUTHOR: frndlytm
