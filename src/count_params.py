import torch
from model import ModularTextModel
from transformers.models.auto.tokenization_auto import AutoTokenizer
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python count_params.py <checkpoint_path>")
        sys.exit(1)
    checkpoint_path = sys.argv[1]
    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = tokenizer.vocab_size
    model = ModularTextModel(
        vocab_size=vocab_size
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params}")
