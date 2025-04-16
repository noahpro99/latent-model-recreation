import os
from datasets import load_dataset
from dotenv import load_dotenv
import torch
from torch.utils.data import Dataset, DataLoader
from transformers.models.auto.tokenization_auto import AutoTokenizer

load_dotenv()


def load_textbooks_dataset(split="train", streaming=False):
    print(f"Loading '{split}' split")
    dataset = load_dataset(
        "nampdn-ai/tiny-strange-textbooks",
        split=split if not streaming else None,
        cache_dir="/mnt/disks/disk-1/cache",
        token=os.getenv("HF_TOKEN"),
        streaming=streaming,
    )
    print(f"Loaded {len(dataset)} examples.")
    return dataset


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
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)
    return batch[:, :-1], batch[:, 1:]


def get_dataloader(batch_size=32, seq_len=64, device="cpu", streaming=True):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = load_textbooks_dataset(streaming=streaming)

    def gen():
        for ex in dataset["train"] if streaming else dataset:
            ids = tokenizer.encode(ex["text"], truncation=True, max_length=seq_len)
            if len(ids) > 1:
                yield torch.tensor(ids)

    dataloader = DataLoader(
        gen(),
        batch_size=batch_size,
        collate_fn=collate_fn,
        pin_memory=(device == "cuda"),
    )
    return dataloader, tokenizer
