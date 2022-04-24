import torch
import torch.nn as nn
from torch.nn import functional as F, init


class SharedHistory(nn.Module):
    """
    A SharedHistory is a Tensor-wrapper module that updates by
    rolling out the oldest memory and concatenating the newest
    memory on the end.
    """
    def __init__(self, tensor: torch.Tensor):
        self.history = nn.Parameter(tensor, requires_grad=False)

    def forward(self, next_: torch.Tensor):
        self.history = torch.concat(self.history[:-1], next_)


class SimpleAttention(nn.Module):
    def __init__(self, d_utter, context: nn.Parameter):
        self.d_utter = d_utter
        self.d_context = context.size(0)
        self.d_attention = context.size(1) - 1

        self.alpha = nn.Parameter(torch.ones(self.d_attention))
        self.W_alpha = nn.Parameter(torch.empty(self.d_utter, self.d_global))
        self.context = context

    def init_weights(self):
        init.xavier_uniform_(
            self.W_alpha, gain=(
                (self.d_utter + self.d_global)
                / torch.abs(self.dutter - self.d_globa)
            )
        )

    def forward(self, utter: torch.Tensor) -> torch.Tensor:
        history = self.context.state[:, :-1]
        self.alpha = F.softmax(utter @ self.W_alpha @ history)
        return self.alpha @ history


class Conversationalist(nn.Module):
    def __init__(
        self,
        d_state: int,
        d_context: int,
        d_utter: int,
        d_emotion: int,
        name: str,
        context: nn.Module,
    ):
        # Shape
        self.d_state = d_state
        self.d_context = d_context
        self.d_utter = d_utter
        self.d_emotion = d_emotion

        # Conversationalist
        self.name = name
        self.context = context
        self.attention = SimpleAttention(d_utter, context)

        self.state = nn.Parameter(torch.zeros(d_state), requires_grad=False)
        self.emotion = nn.Parameter(torch.zeros(d_emotion), requires_grad=False)
        self.emote = nn.GRU(d_emotion, d_state)

        # Roles
        self.as_speaker = nn.GRU(d_state, d_utter + d_context)
        self.as_listener = nn.GRU(d_state, d_utter + d_context)

    def listen(self, utter: torch.Tensor, context: torch.Tensor):
        """listen(...) updates the conversationalist internal state by the utter
        and the current global context as if they spoke it."""
        self.state = self.as_listener(self.state, torch.concat(utter, context))

    def speak(self, utter: torch.Tensor, context: torch.Tensor):
        """speak(...) updates the conversationalist internal state by the utter
        and the current global context as if they spoke it."""
        self.state = self.as_speaker(self.state, torch.concat(utter, context))

    def forward(self, speaker: str, utter: torch.Tensor) -> torch.Tensor:
        """forward through the conversation, receives the speaker and its
        utterance to update the current state, and emit an emotion"""
        context = self.context.history[-1]
        if self.name == speaker:
            self.speak(utter, context)
        else:
            self.listen(utter, context)

        self.emotion = self.emote(self.emotion, self.state)
        return self.emotion


class DialgueRNN(torch.nn.Module):
    def __init__(
        self,
        parties: set[str],
        d_party: int,
        d_utter: int,
        d_emotion: int,
        d_context: int,
        d_attention: int,
    ):
        self.d_party = d_party
        self.d_utter = d_utter
        self.d_emotion = d_emotion
        self.d_context = d_context
        self.d_attention = d_attention

        # Configure global conversation state module and an empty history
        # of size d_history representing how many utterances we want to
        # pay long-term attention to.
        self.global_gru = nn.GRU(d_utter, d_context)
        self.global_context = SharedHistory(torch.zeros(d_context, d_attention))

        # Track the state of who's talking and who's listening
        # in independent Conversationalists
        self.parties = torch.nn.ModuleDict({
            name: Conversationalist(
                d_party=d_party,
                d_utter=d_utter,
                d_context=d_context,
                name=name,
                context=self.global_context,
            )
            for name in parties
        })

    def forward(self, speaker: str, utter: torch.tensor):
        """
        Returns
        -------
            torch.Tensor
                The stacked emotion predictions of each Conversationalist
                in the dialogue
        """
        out = torch.empty(self.d_emotion)

        # Allow each party to handle the utterance
        for name, party in self.parties.items():
            emotion = party(speaker, utter)

            # Copy the speaker's emotion to the output
            if speaker == name:
                out.copy_(emotion)

        # RETURN
        return out
