import torch
import torch.nn as nn


class InputBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.linear(x))


class RecurrentDecodeBlock(nn.Module):
    def __init__(self, hidden_dim, recurrence=3):
        super().__init__()
        self.recurrence = recurrence
        self.block = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())

    def forward(self, x):
        outputs = []
        for _ in range(self.recurrence):
            x = self.block(x)
            outputs.append(x)
        return torch.stack(outputs, dim=0)  # [recurrence, batch, dim]


class FinalTokenBlock(nn.Module):
    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        return self.linear(x)


class ModularTextModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size, recurrence):
        super().__init__()
        self.decode = InputBlock(input_dim, hidden_dim)
        self.recurrent = RecurrentDecodeBlock(hidden_dim, recurrence)
        self.final = FinalTokenBlock(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.decode(x)
        rec_outs = self.recurrent(x)  # [recurrence, batch, dim]
        final_outs = self.final(rec_outs)  # [recurrence, batch, vocab]
        return final_outs
