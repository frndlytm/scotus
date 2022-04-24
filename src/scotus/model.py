import torch
import torch.nn as nn
from torch.nn import functional as F, init


class SharedMemory(nn.Module):
    """
    A SharedHistory is a Tensor-wrapper module that updates by
    rolling out the oldest memory and concatenating the newest
    memory on the end.
    """
    def __init__(self, tensor: torch.Tensor):
        super().__init__()
        self.state = nn.Parameter(tensor, requires_grad=False)

    def history(self):
        return self.state[:-1]

    def now(self):
        return self.state[-1]

    def forward(self, next_: torch.Tensor):
        self.state = torch.concat(self.state[:-1], next_)


class SimpleAttention(nn.Module):
    def __init__(self, d_utter, context: SharedMemory):
        super().__init__()

        # Extract attention dimensions from the shared memory
        self.context = context
        self.d_utter = d_utter
        self.d_context = self.context.size(1)
        self.d_attention = self.context.size(0) - 1

        # Learned attention parameters
        self.alpha = nn.Parameter(torch.ones(self.d_attention))
        self.W_alpha = nn.Parameter(torch.empty(self.d_utter, self.d_global))
        init.xavier_uniform_(
            self.W_alpha, gain=(
                (self.d_utter + self.d_global)
                / torch.abs(self.dutter - self.d_globa)
            )
        )

    def forward(self, utter: torch.Tensor) -> torch.Tensor:
        history = self.context.history()
        self.alpha = F.softmax(utter @ self.W_alpha @ history)
        return self.alpha @ history


class Conversationalist(nn.Module):
    """
    A Conversationalist is a member of a Conversation that shares conversation
    context with other Conversationalists. A Conversationalist has a `name`
    that helps to distinguish if it is speaking or listening.
    """
    def __init__(
        self,
        d_state: int,
        d_context: int,
        d_utter: int,
        d_emotion: int,
        name: str,
        context: SharedMemory,
    ):
        super().__init__()

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

    def reset(self):
        init.zeros_(self.state)
        init.zeros_(self.emotion)

    def listen(self, utter: torch.Tensor, context: torch.Tensor):
        """listen(...) updates the conversationalist internal state by the utter
        and the current global context as if they spoke it."""
        self.as_listener(self.state, torch.concat(utter, context)).copy_(self.state)

    def speak(self, utter: torch.Tensor, context: torch.Tensor):
        """speak(...) updates the conversationalist internal state by the utter
        and the current global context as if they spoke it."""
        self.as_speaker(self.state, torch.concat(utter, context)).copy_(self.state)

    def forward(self, speaker: str, utter: torch.Tensor) -> torch.Tensor:
        """forward through the conversation, receives the speaker and its
        utterance to update the current state, and emit an emotion"""
        context = self.context.now()

        if self.name == speaker:
            self.speak(utter, context)
        else:
            self.listen(utter, context)

        self.emote(self.emotion, self.state).copy_(self.emotion)
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
        super().__init__()
        self.d_party = d_party
        self.d_utter = d_utter
        self.d_emotion = d_emotion
        self.d_context = d_context
        self.d_attention = d_attention

        # Configure global conversation state module and an empty history
        # of size d_attention representing how many contexts we want to
        # pay attention to.
        self.global_gru = nn.GRU(d_utter, d_context)
        self.global_context = SharedMemory(torch.zeros(d_context, d_attention))

        # Track the state of who's talking and who's listening
        # in independent Conversationalists
        self.parties = list(parties)
        self.conversationalists = torch.nn.ModuleDict({
            name: Conversationalist(
                d_party=d_party,
                d_utter=d_utter,
                d_context=d_context,
                name=name,
                context=self.global_context,
            )
            for name in self.parties
        })

        # TODO:
        #     TITLE: Multi-Conversation Conversationalists
        #     AUTHOR: frndlytm
        #     DESCRIPTION:
        #
        #         Suppose a Conversationalist is involved in multiple conversations,
        #         i.e. multiple comment threads on a social network. We can manage
        #         multiple conversation contexts as memories as well.
        #
        #         Perhaps, DialogueRNN is at the Conversation-level; then consider
        #         how to share speakers among Conversations?
        #
        self.conversation = None

    def set_conversation(self, conversation: str):
        # if we have started a new conversation
        if self.conversation != conversation:
            self.conversation = conversation

            # In new conversations, the party state should
            # be reset; however, they should remember what
            # they have learned (to believe), and still
            # leverage the global context since it hasn't
            # changed
            for party in self.parties.values():
                party.reset_state()

    def forward(
        self,
        conversation: torch.Tensor,
        speaker: torch.Tensor,
        utter: torch.Tensor,
    ):
        """
        Parameters
        ----------
            conversation: torch.Tensor
            speaker: torch.Tensor
            utter: torch.Tensor

        Returns
        -------
            torch.Tensor
                The stacked emotion predictions of each Conversationalist
                in the dialogue
        """
        # since the speaker tensor is a one-hot encoding of the speakers
        # in the conversation, argmax returns the index of the speaker name
        self.set_conversation(conversation)
        speaker_ = self.parties[torch.argmax(speaker)]

        # Allow each party to handle the utterance and emit an `emotion` tensor
        out = torch.empty(self.d_emotion)
        for name, converse in self.conversationalists.items():
            emotion = converse(speaker, utter)

            # Only copy the speaker's emotion to the output
            if speaker_ == name:
                out.copy_(emotion)

        # RETURN
        return out
