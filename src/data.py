import os

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers.models.auto.tokenization_auto import AutoTokenizer

load_dotenv()


def load_textbooks_dataset(split="train", streaming=False):
    print(f"Loading '{split}'" if not streaming else "Loading streaming")
    dataset = load_dataset(
        "nampdn-ai/tiny-strange-textbooks",
        split=split if not streaming else None,
        cache_dir=os.path.join(os.getenv("CACHE_DIR")),
        token=os.getenv("HF_TOKEN"),
        streaming=streaming,
    )
    print(f"Loaded {len(dataset)} examples.")
    return dataset


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_len=64):
        self.examples = []
        self.pad_token_id = tokenizer.pad_token_id
        for t in texts:
            ids = tokenizer.encode(t, truncation=True, max_length=None)
            n = len(ids)
            if n < 1:
                continue
            # Sliding window: create all possible chunks of length seq_len
            for i in range(n):
                chunk = ids[max(0, i - seq_len + 1) : i + 1]
                # Left-pad to seq_len
                if len(chunk) < seq_len:
                    chunk = [self.pad_token_id] * (seq_len - len(chunk)) + chunk
                self.examples.append(torch.tensor(chunk))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class StreamingTextDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, seq_len=64):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.pad_token_id = tokenizer.pad_token_id

    def __iter__(self):
        for ex in self.dataset["train"]:
            ids = self.tokenizer.encode(ex["text"], truncation=True, max_length=None)
            n = len(ids)
            if n < 1:
                continue
            # Sliding window: create all possible chunks of length seq_len
            for i in range(n):
                chunk = ids[max(0, i - self.seq_len + 1) : i + 1]
                # Left-pad to seq_len
                if len(chunk) < self.seq_len:
                    chunk = [self.pad_token_id] * (self.seq_len - len(chunk)) + chunk
                yield torch.tensor(chunk)


def collate_fn(batch):
    # batch: [batch, seq_len]
    batch = torch.stack(batch, dim=0)
    # Input: all but last token, Target: all but first token
    x = batch[:, :-1]
    y = batch[:, -1]
    return x, y


def get_dataloader(batch_size=32, seq_len=64, device="cpu", streaming=True):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = load_textbooks_dataset(split="train[:1000]", streaming=streaming)
    if streaming:
        ds = StreamingTextDataset(dataset, tokenizer, seq_len=seq_len)
    else:
        texts = [ex["text"] for ex in dataset]
        ds = TextDataset(texts, tokenizer, seq_len=seq_len)
    dataloader = DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=collate_fn,
        pin_memory=(device == "cuda"),
        shuffle=not streaming,
    )
    return dataloader, tokenizer


if __name__ == "__main__":
    dataloader, tokenizer = get_dataloader(batch_size=2, seq_len=16, streaming=True)
    data_iter = iter(dataloader)

    print(
        "Streaming mode interactive demo. Press Enter to get the next batch (Ctrl+C to exit)."
    )
    while True:
        input()
        try:
            x, y = next(data_iter)
        except StopIteration:
            print("End of dataset. Restarting iterator.")
            data_iter = iter(dataloader)
            x, y = next(data_iter)
        print("Encoded input (x):", x)
        print(
            "Decoded input (x):",
            [tokenizer.decode(seq, skip_special_tokens=True) for seq in x],
        )
        print("Encoded target (y):", y)
        print(
            "Decoded target (y):",
            [tokenizer.decode(seq, skip_special_tokens=True) for seq in y],
        )
