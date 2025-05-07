import torch
from transformers.models.auto.tokenization_auto import AutoTokenizer
from model import ModularTextModel


def manual_test(checkpoint=None, seq_len=64):
    print("[DEBUG] Starting manual_test")
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEBUG] Using device: {device}")

    input_text = input("Enter text to test the model: ")
    print(f"[DEBUG] User input: {input_text}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = tokenizer.vocab_size
    print(f"[DEBUG] Tokenizer loaded. Vocab size: {vocab_size}")

    # Tokenize input
    tokens = tokenizer.encode(input_text, add_special_tokens=True)
    print(f"[DEBUG] Tokenized input: {tokens}")
    # Pad input to seq_len
    if len(tokens) < seq_len:
        tokens = tokens + [tokenizer.pad_token_id] * (seq_len - len(tokens))
    else:
        tokens = tokens[:seq_len]
    x = torch.tensor(tokens).unsqueeze(0).to(device)  # [1, seq_len]
    print(f"[DEBUG] Input tensor shape: {x.shape}")

    # Model
    model = ModularTextModel(
        input_dim=vocab_size, hidden_dim=128, vocab_size=vocab_size, recurrence=3
    ).to(device)
    print("[DEBUG] Model instantiated")
    if checkpoint is not None:
        print(f"[DEBUG] Loading checkpoint: {checkpoint}")
        checkpoint_data = torch.load(checkpoint, map_location=device)
        model.load_state_dict(checkpoint_data["model_state"])
        print(f"Loaded checkpoint from {checkpoint}")

    model.eval()
    with torch.no_grad():
        generated = x.clone()
        orig_seq_len = x.shape[1]
        max_gen_len = 200
        print(f"[DEBUG] Generating up to {max_gen_len - orig_seq_len} new tokens")
        for _ in range(orig_seq_len, max_gen_len):
            if generated.shape[1] < seq_len:
                input_seq = torch.cat([
                    generated,
                    torch.full((1, seq_len - generated.shape[1]), tokenizer.pad_token_id, device=device, dtype=generated.dtype)
                ], dim=1)
            else:
                input_seq = generated[:, -seq_len:]
            logits = model(input_seq)  # [batch, vocab]
            next_token_logits = logits[0]  # [vocab]

            # Mask pad token so it cannot be chosen
            next_token_logits[tokenizer.pad_token_id] = float('-inf')

            probs = torch.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
            print(f"[DEBUG] Next token id: {next_token_id.item()}")
            if next_token_id.item() == tokenizer.sep_token_id or next_token_id.item() == tokenizer.eos_token_id:
                print("[DEBUG] Stopping generation: reached SEP or EOS token.")
                break
            generated = torch.cat([generated, next_token_id.unsqueeze(0)], dim=1)

        predicted_ids = generated[0].cpu().numpy().tolist()
        output_text = tokenizer.decode(predicted_ids, skip_special_tokens=True)
        print(output_text)
