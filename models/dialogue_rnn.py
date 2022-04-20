import torch
import torch.nn.funtional as F
from torch import nn

from .gru import GRU


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
        parties: set[str],
        d_global: int,
        d_party: int,
        d_utter: int,
        d_history: int,
        speaker: Type[nn.Module] = SpeakerGRU,
        listener: nn.Module = PartyGRU,
    ):
        # Track the state of who's talking and who's listening
        self.channels = len(parties)
        self.is_speaker = torch.zeros(self.channels)

        # Configure parties' individual hidden contexts
        self.party = {p: i for i, p in enumerate(parties)}
        self.party_state = nn.Parameter(torch.zeros(self.channels, d_party))
        self.party_grus = [
            SpeakerGRU(name, d_global, d_utter, d_history)
            for name in parties
        ]

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
        