from datetime import datetime
import itertools
import os

import torch
import torch.nn as nn
import torch.optim as optim

from data import get_dataloader
from model import ModularTextModel
from transformers.models.auto.tokenization_auto import AutoTokenizer
from tqdm import trange, tqdm


def train(
    checkpoint=None,
    epochs=3,
    batch_size=32,
    checkpoint_path=f"checkpoints/checkpoint-{datetime.now().strftime('%Y%m%d-%H%M%S')}.pt",
):
    os.makedirs("./checkpoints", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    loader, tokenizer = get_dataloader(
        batch_size=batch_size, device=str(device), streaming=True
    )
    vocab_size = tokenizer.vocab_size
    model = ModularTextModel(vocab_size=vocab_size).to(device)
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
    max_batches = 1000
    for epoch in trange(start_epoch, start_epoch + epochs, desc="Epochs", disable=True):
        print(f"Epoch {epoch+1}/{start_epoch + epochs}")
        for batch_i, (x, y) in enumerate(itertools.islice(loader, max_batches)):
            x = x.to(torch.long).to(device)
            y = y.to(torch.long).to(device)
            out = model(x)  # x is token indices
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(
                f"Epoch {epoch+1}, Batch {batch_i+1}/{max_batches}, Loss: {loss.item():.4f}",
                end="\r",
            )
        print(f"Epoch {epoch+1} done. Loss: {loss.item():.4f}")
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch + 1,
            },
            checkpoint_path,
        )
        print(f"Checkpoint saved to {checkpoint_path}")
