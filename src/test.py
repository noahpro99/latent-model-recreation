import torch
from transformers.models.auto.tokenization_auto import AutoTokenizer
from train import ModularTextModel


def manual_test(checkpoint=None):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    input_text = input("Enter text to test the model: ")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = tokenizer.vocab_size

    # Tokenize input
    tokens = tokenizer.encode(input_text, add_special_tokens=True)
    x = torch.tensor(tokens).unsqueeze(0).to(device)  # [1, seq_len]

    # Model
    model = ModularTextModel(
        input_dim=vocab_size, hidden_dim=128, vocab_size=vocab_size, recurrence=3
    ).to(device)
    if checkpoint is not None:
        checkpoint_data = torch.load(checkpoint, map_location=device)
        model.load_state_dict(checkpoint_data["model_state"])
        print(f"Loaded checkpoint from {checkpoint}")

    model.eval()
    with torch.no_grad():
        # Embed tokens
        emb = torch.nn.functional.one_hot(x, num_classes=vocab_size).float()
        outs = model(emb)  # [recurrence, batch, seq, vocab]
        # Take the last recurrence output
        logits = outs[-1, 0]  # [seq, vocab]
        predicted_ids = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
        # Decode predicted tokens to text
        output_text = tokenizer.decode(predicted_ids, skip_special_tokens=True)
        print("Input text: ", input_text)
        print("Output text:", output_text)
