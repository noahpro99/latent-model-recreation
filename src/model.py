import torch
import torch.nn as nn


class InputBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.linear(x))


class RecurrentStackBlock(nn.Module):
    def __init__(self, hidden_dim, num_layers=4, num_heads=24, ff_mult=4, dropout=0.1, num_recurrences=24):
        super().__init__()
        self.num_recurrences = num_recurrences
        self.layers = nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(hidden_dim),
                nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True),
                nn.LayerNorm(hidden_dim),
                nn.Sequential(
                    nn.Linear(hidden_dim, ff_mult * hidden_dim),
                    nn.GELU(),
                    nn.Linear(ff_mult * hidden_dim, hidden_dim),
                    nn.Dropout(dropout),
                )
            ]) for _ in range(num_layers)
        ])

    def forward(self, x):
        for _ in range(self.num_recurrences):
            for norm1, attn, norm2, ff in self.layers:
                x_norm = norm1(x)
                attn_out, _ = attn(x_norm, x_norm, x_norm, need_weights=False)
                x = x + attn_out
                x_norm = norm2(x)
                ff_out = ff(x_norm)
                x = x + ff_out
        return x


class FinalTokenBlock(nn.Module):
    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        return self.linear(x)


class ModularTextModel(nn.Module):
    def __init__(self, hidden_dim=360, vocab_size=None, num_layers=4, num_heads=24, ff_mult=4, dropout=0.1, num_recurrences=24):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.decode = InputBlock(hidden_dim)
        self.recurrent = RecurrentStackBlock(hidden_dim, num_layers=num_layers, num_heads=num_heads, ff_mult=ff_mult, dropout=dropout, num_recurrences=num_recurrences)
        self.final = FinalTokenBlock(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.decode(x)
        x = self.recurrent(x)
        x = self.final(x)
        return x[:, -1, :]
