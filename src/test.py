import csv
import os
from matplotlib import pyplot as plt
import torch
from tqdm import tqdm

from data import get_dataloader
from model import RecurrentTransformerModel


def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
        # Crop current context if it exceeds the supported context size
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


def generate_and_print_sample(model, tokenizer, device, start_context, seq_len=64):
    model.eval()
    encoded = tokenizer.encode(
        start_context, add_special_tokens=True, truncation=True, max_length=seq_len
    )
    # Convert to tensor and add batch dimension
    encoded_tensor = torch.tensor([encoded], device=device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded_tensor, max_new_tokens=50, context_size=seq_len
        )
    decoded_text = tokenizer.decode(token_ids[0], skip_special_tokens=True)
    print(decoded_text.replace("\n", " "))
    model.train()


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
    loader, tokenizer = get_dataloader(batch_size=batch_size, num_workers=4)
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
            model = RecurrentTransformerModel(
                vocab_size=vocab_size, num_recurrences=recurrence
            ).to(device)

            # Load checkpoint weights
            checkpoint_data = torch.load(checkpoint, map_location=device)
            model.load_state_dict(checkpoint_data["model_state"], strict=False)

            # Set model to eval mode
            model.eval()

            # Set recurrence level
            model.num_recurrences = recurrence

            # Calculate loss on samples
            criterion = torch.nn.CrossEntropyLoss()
            total_loss = 0.0
            samples_processed = 0

            with torch.no_grad():
                data_iter = iter(loader)
                pbar = tqdm(total=num_samples, desc=f"Recurrence {recurrence}")

                while samples_processed < num_samples:
                    try:
                        inputs, targets = next(data_iter)
                    except StopIteration:
                        data_iter = iter(loader)
                        inputs, targets = next(data_iter)

                    inputs = inputs.to(torch.long).to(device)
                    targets = targets.to(torch.long).to(device)

                    # Forward pass
                    logits = model(inputs)  # [batch, seq_len, vocab]

                    # Reshape for loss calculation
                    batch_size, seq_len = targets.size()
                    logits_flat = logits.view(-1, vocab_size)
                    targets_flat = targets.view(-1)

                    # Calculate loss
                    loss = criterion(logits_flat, targets_flat)

                    # Update statistics
                    batch_samples = inputs.size(0)
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
