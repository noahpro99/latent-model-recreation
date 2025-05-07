import torch
import csv
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers.models.auto.tokenization_auto import AutoTokenizer
from model import ModularTextModel
from data import get_dataloader


def manual_test(checkpoint=None, seq_len=64):
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
    model = ModularTextModel(vocab_size=vocab_size).to(device)
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
        max_gen_len = seq_len * 2
        print(f"[DEBUG] Generating up to {max_gen_len - orig_seq_len} new tokens")
        for _ in range(orig_seq_len, max_gen_len):
            if generated.shape[1] < seq_len:
                input_seq = torch.cat(
                    [
                        generated,
                        torch.full(
                            (1, seq_len - generated.shape[1]),
                            tokenizer.pad_token_id,
                            device=device,
                            dtype=generated.dtype,
                        ),
                    ],
                    dim=1,
                )
            else:
                input_seq = generated[:, -seq_len:]
            logits = model(input_seq)  # [batch, vocab]
            next_token_logits = logits[0]  # [vocab]

            probs = torch.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
            if (
                next_token_id.item() == tokenizer.sep_token_id
                or next_token_id.item() == tokenizer.eos_token_id
            ):
                break
            generated = torch.cat([generated, next_token_id.unsqueeze(0)], dim=1)

        predicted_ids = generated[0].cpu().numpy().tolist()
        output_text = tokenizer.decode(predicted_ids, skip_special_tokens=True)
        print(output_text)


def evaluate_recurrence_levels(
    checkpoint=None,
    recurrence_levels=[1, 2, 4, 6, 8, 12, 24],
    num_samples=500,
    batch_size=32,
    output_dir="./evaluation",
):
    """
    Evaluates model performance at different recurrence levels and generates a CSV and plot.

    Args:
        checkpoint: Path to the model checkpoint
        recurrence_levels: List of recurrence levels to evaluate
        num_samples: Number of examples to evaluate on
        batch_size: Batch size for evaluation
        output_dir: Directory to save CSV and plot
    """
    os.makedirs(output_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    loader, tokenizer = get_dataloader(
        batch_size=batch_size, device=str(device), streaming=True
    )
    vocab_size = tokenizer.vocab_size

    # Load base model from checkpoint
    if checkpoint is None or not os.path.isfile(checkpoint):
        print("No valid checkpoint provided. Please provide a valid checkpoint.")
        return

    # Prepare CSV file
    csv_path = os.path.join(output_dir, "recurrence_evaluation.csv")
    with open(csv_path, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Recurrence_Level", "Loss"])

        results = []

        # Evaluate each recurrence level
        for recurrence in recurrence_levels:
            print(f"Evaluating model with recurrence level {recurrence}")

            # Create model with current recurrence level
            model = ModularTextModel(
                vocab_size=vocab_size, num_recurrences=recurrence
            ).to(device)

            # Load checkpoint weights
            checkpoint_data = torch.load(checkpoint, map_location=device)
            model.load_state_dict(checkpoint_data["model_state"], strict=False)

            # Set model to eval mode
            model.eval()

            # Set recurrence level
            model.recurrent.num_recurrences = recurrence

            # Calculate loss on samples
            criterion = torch.nn.CrossEntropyLoss()
            total_loss = 0.0
            samples_processed = 0

            with torch.no_grad():
                data_iter = iter(loader)
                pbar = tqdm(total=num_samples, desc=f"Recurrence {recurrence}")

                while samples_processed < num_samples:
                    try:
                        x, y = next(data_iter)
                    except StopIteration:
                        data_iter = iter(loader)
                        x, y = next(data_iter)

                    x = x.to(torch.long).to(device)
                    y = y.to(torch.long).to(device)

                    # Forward pass
                    out = model(x)
                    loss = criterion(out, y)

                    # Update statistics
                    batch_samples = x.size(0)
                    samples_to_add = min(batch_samples, num_samples - samples_processed)
                    total_loss += loss.item() * samples_to_add
                    samples_processed += samples_to_add
                    pbar.update(samples_to_add)

                    if samples_processed >= num_samples:
                        break

            # Calculate average loss
            avg_loss = total_loss / num_samples
            print(f"Recurrence level {recurrence} - Average Loss: {avg_loss:.4f}")

            # Write to CSV
            csv_writer.writerow([recurrence, avg_loss])
            results.append((recurrence, avg_loss))

    print(f"Evaluation results saved to {csv_path}")

    # Create plot
    recurrences, losses = zip(*results)
    plt.figure(figsize=(10, 6))
    plt.plot(recurrences, losses, marker="o", linestyle="-")
    plt.title("Model Loss vs Recurrence Level")
    plt.xlabel("Recurrence Level")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.xticks(recurrences)

    plot_path = os.path.join(output_dir, "recurrence_plot.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    plt.close()

    return results
