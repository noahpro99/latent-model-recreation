import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads=12, ff_mult=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_mult * hidden_dim),
            nn.GELU(),
            nn.Linear(ff_mult * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.drop_shortcut = nn.Dropout(dropout)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(
            x.device
        )
        attn_out, _ = self.attn(x, x, x, attn_mask=causal_mask, is_causal=True)
        x = self.drop_shortcut(attn_out)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        ff_out = self.ff(x)
        x = self.drop_shortcut(ff_out)
        x = x + shortcut
        return x


class RecurrentTransformerModel(nn.Module):
    def __init__(
        self,
        hidden_dim=768,
        vocab_size=50257,
        num_recurrent_layers=8,
        num_heads=12,
        ff_mult=4,
        dropout=0.1,
        num_recurrences=1,
        seq_len=1024,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(seq_len, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.input_transformer = TransformerBlock(
            hidden_dim, num_heads, ff_mult, dropout
        )

        self.recurrent_layers = nn.ModuleList(
            [
                TransformerBlock(hidden_dim, num_heads, ff_mult, dropout)
                for _ in range(num_recurrent_layers)
            ]
        )
        self.num_recurrences = num_recurrences

        self.output_transformer = TransformerBlock(
            hidden_dim, num_heads, ff_mult, dropout
        )

        self.final_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, x):
        b, seq_len = x.shape

        x = self.embedding(x)

        x = x + self.pos_embedding(
            torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        )
        x = self.dropout(x)

        x = self.input_transformer(x)

        for _ in range(self.num_recurrences):
            for layer in self.recurrent_layers:
                x = layer(x)

        x = self.output_transformer(x)

        x = self.final_norm(x)
        logits = self.head(x)

        return logits
