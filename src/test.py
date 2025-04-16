import torch
from transformers import AutoTokenizer
from train import ModularTextModel, TextDataset, collate_fn
from load_dataset_pretrain import load_textbooks_dataset
from torch.utils.data import DataLoader


def test(checkpoint=None, batch_size=8):
    # Load dataset and tokenizer
    dataset = load_textbooks_dataset(save_to_file=False)
    texts = [ex["text"] for ex in dataset]
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = tokenizer.vocab_size
    ds = TextDataset(texts, tokenizer)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Model
    model = ModularTextModel(
        input_dim=vocab_size, hidden_dim=128, vocab_size=vocab_size, recurrence=3
    )
    if checkpoint is not None:
        checkpoint_data = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint_data["model_state"])
        print(f"Loaded checkpoint from {checkpoint}")

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(torch.long)
            # Embed tokens
            emb = torch.nn.functional.one_hot(x, num_classes=vocab_size).float()
            outs = model(emb)  # [recurrence, batch, seq, vocab]
            print(f"Model output logits shape: {outs.shape}")
            print(
                f"Sample logits: {outs[0,0,0,:5]}"
            )  # Print first 5 logits of first token
            break  # Only run one batch for basic test
