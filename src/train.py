import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from load_dataset_pretrain import load_textbooks_dataset


# --- Modular Blocks ---
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


# --- Main Model ---
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


# --- Dataset Wrapper ---
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_len=64):
        self.examples = []
        for t in texts:
            ids = tokenizer.encode(t, truncation=True, max_length=seq_len)
            if len(ids) > 1:
                self.examples.append(torch.tensor(ids))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch):
    batch = nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)
    return batch[:, :-1], batch[:, 1:]


# --- Training Loop ---
def train():
    # Load dataset
    dataset = load_textbooks_dataset(save_to_file=False)
    texts = [ex["text"] for ex in dataset]
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = tokenizer.vocab_size
    ds = TextDataset(texts, tokenizer)
    loader = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=collate_fn)

    # Model
    model = ModularTextModel(
        input_dim=64, hidden_dim=128, vocab_size=vocab_size, recurrence=3
    )
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(3):
        for x, y in loader:
            x = x.to(torch.long)
            y = y.to(torch.long)
            # Embed tokens
            emb = nn.functional.one_hot(x, num_classes=vocab_size).float()
            outs = model(emb)  # [recurrence, batch, seq, vocab]
            loss = 0
            for out in outs:
                # out: [batch, seq, vocab]
                loss += criterion(out.view(-1, vocab_size), y.view(-1))
            loss = loss / outs.shape[0]  # Average over recurrence
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} done. Loss: {loss.item():.4f}")


if __name__ == "__main__":
    train()
