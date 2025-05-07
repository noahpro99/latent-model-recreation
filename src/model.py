import torch
import torch.nn as nn


class InputBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.linear(x))


class RecurrentDecodeBlock(nn.Module):
    def __init__(self, hidden_dim, recurrence=3):
        super().__init__()
        self.recurrence = recurrence
        self.block = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())

    def forward(self, x):
        for _ in range(self.recurrence):
            x = self.block(x)
        return x  # Only return the final output


class FinalTokenBlock(nn.Module):
    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        return self.linear(x)


class ModularTextModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size, recurrence):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.decode = InputBlock(hidden_dim)
        self.recurrent = RecurrentDecodeBlock(hidden_dim, recurrence)
        self.final = FinalTokenBlock(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)  # [batch, seq, hidden_dim]
        x = self.decode(x)
        x = self.recurrent(x)  # [batch, seq, hidden_dim]
        x = self.final(x)      # [batch, seq, vocab]
        return x[:, -1, :]  # Only return the last token's output for each sequence [batch, vocab]
