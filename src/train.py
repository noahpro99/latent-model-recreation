from datetime import datetime
import itertools
import os

import torch
import torch.nn as nn
import torch.optim as optim

from data import get_dataloader
from model import RecurrentTransformerModel
from tqdm import trange

from test import generate_and_print_sample, generate_text_simple


def train(
    checkpoint=None,
    epochs=3,
    batch_size=32,
    checkpoint_path=f"checkpoints/checkpoint-{datetime.now().strftime('%Y%m%d-%H%M%S')}.pt",
):
    os.makedirs("./checkpoints", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    loader, tokenizer = get_dataloader(batch_size=batch_size, num_workers=4)
    vocab_size = tokenizer.vocab_size
    model = RecurrentTransformerModel(vocab_size=vocab_size, seq_len=128).to(device)
    print(f"Model number of parameters: {sum(p.numel() for p in model.parameters())}")
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
    batch_idx = 0
    max_batches = 100
    for epoch in trange(start_epoch, start_epoch + epochs, desc="Epochs", disable=True):
        for batch_i, (x, y) in enumerate(itertools.islice(loader, max_batches)):
            optimizer.zero_grad()
            x = x.to(torch.long).to(device)
            y = y.to(torch.long).to(device)
            out = model(x)
            loss = criterion(out.view(-1, out.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            batch_idx += 1
            print(
                f"Epoch {epoch+1}. Batch : {batch_idx} Loss: {loss.item():.4f}",
                end="\r",
            )
        print()
        generate_and_print_sample(
            model,
            tokenizer,
            device,
            start_context="Introduction: ",
            seq_len=128,
        )
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch + 1,
            },
            checkpoint_path,
        )
        print(f"Checkpoint saved to {checkpoint_path}")
