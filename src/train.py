from datetime import datetime
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from data import load_textbooks_dataset


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


def train(
    checkpoint=None,
    epochs=3,
    batch_size=32,
    checkpoint_path=f"checkpoints/checkpoint-{datetime.now().strftime('%Y%m%d-%H%M%S')}.pt",
):
    os.makedirs("./checkpoints", exist_ok=True)
    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Load dataset
    dataset = load_textbooks_dataset(save_to_file=False)
    texts = [ex["text"] for ex in dataset]
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = tokenizer.vocab_size
    ds = TextDataset(texts, tokenizer)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Model
    model = ModularTextModel(
        input_dim=vocab_size, hidden_dim=128, vocab_size=vocab_size, recurrence=3
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    start_epoch = 0

    # Optionally load checkpoint if provided
    if checkpoint is not None and os.path.isfile(checkpoint):
        checkpoint_data = torch.load(checkpoint, map_location=device)
        model.load_state_dict(checkpoint_data["model_state"])
        optimizer.load_state_dict(checkpoint_data["optimizer_state"])
        start_epoch = checkpoint_data.get("epoch", 0)
        print(f"Loaded checkpoint from {checkpoint}, starting at epoch {start_epoch+1}")

    model.train()

    for epoch in range(start_epoch, epochs):
        for x, y in loader:
            x = x.to(torch.long).to(device)
            y = y.to(torch.long).to(device)
            # Embed tokens
            emb = nn.functional.one_hot(x, num_classes=vocab_size).float().to(device)
            outs = model(emb)  # [recurrence, batch, seq, vocab]
            loss = 0
            for out in outs:
                # out: [batch, seq, vocab]
                loss += criterion(out.reshape(-1, vocab_size), y.reshape(-1))
            loss = loss / outs.shape[0]  # Average over recurrence
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} done. Loss: {loss.item():.4f}")

        # Save checkpoint after each epoch
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch + 1,
            },
            checkpoint_path,
        )
        print(f"Checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    train()
