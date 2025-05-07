import torch
from model import ModularTextModel
from transformers.models.auto.tokenization_auto import AutoTokenizer
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def count_params(checkpoint_path):
    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = tokenizer.vocab_size
    model = ModularTextModel(vocab_size=vocab_size)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params}")


def plot_recurrence_evaluation(csv_path, output_dir="evaluation"):
    """
    Plots the recurrence evaluation results from a CSV file.

    Args:
        csv_path: Path to the CSV file containing evaluation results
        output_dir: Directory to save the plot
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Read the CSV data
    df = pd.read_csv(csv_path)

    # Set up the plot style
    sns.set_style("whitegrid")
    plt.figure(figsize=(14, 8))

    # Create the plot with gradient color
    ax = plt.subplot(111)
    cmap = plt.cm.viridis
    colors = cmap(df["Epoch"] / df["Epoch"].max())

    # Plot main line and scatter points
    ax.plot(
        df["Epoch"], df["Recurrence_Loss"], "-", linewidth=2, alpha=0.7, color="#1f77b4"
    )
    scatter = ax.scatter(
        df["Epoch"],
        df["Recurrence_Loss"],
        c=df["Epoch"],
        cmap=cmap,
        s=80,
        alpha=0.7,
        edgecolor="white",
        linewidth=0.5,
    )

    # Add smoothed trendline
    if len(df) > 5:  # Only add trendline if we have enough data points
        window_size = min(10, len(df) // 4)
        rolling_mean = (
            df["Recurrence_Loss"].rolling(window=window_size, center=True).mean()
        )
        ax.plot(
            df["Epoch"],
            rolling_mean,
            "r--",
            linewidth=2.5,
            label=f"Rolling Average (window={window_size})",
        )

    # Calculate percent decrease from start to end
    first_loss = df["Recurrence_Loss"].iloc[0]
    last_loss = df["Recurrence_Loss"].iloc[-1]
    percent_decrease = ((first_loss - last_loss) / first_loss) * 100

    # Set plot title and labels with better styling
    plt.title("Recurrence Loss Over Time", fontsize=18, fontweight="bold", pad=20)
    plt.xlabel("Epochs", fontsize=14, labelpad=10)
    plt.ylabel("Loss", fontsize=14, labelpad=10)

    # Add annotations
    plt.annotate(
        f"Start: {first_loss:.4f}",
        xy=(df["Epoch"].iloc[0], first_loss),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3),
    )

    plt.annotate(
        f"End: {last_loss:.4f}\n({percent_decrease:.1f}% decrease)",
        xy=(df["Epoch"].iloc[-1], last_loss),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="green", alpha=0.3),
    )

    # Find the minimum loss point
    min_idx = df["Recurrence_Loss"].idxmin()
    min_epoch = df["Epoch"].iloc[min_idx]
    min_loss = df["Recurrence_Loss"].iloc[min_idx]

    # Mark the minimum loss point
    plt.annotate(
        f"Min: {min_loss:.4f} (Epoch {min_epoch})",
        xy=(min_epoch, min_loss),
        xytext=(0, -25),
        textcoords="offset points",
        fontsize=10,
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
        bbox=dict(boxstyle="round,pad=0.3", fc="cyan", alpha=0.3),
    )

    # Customize the grid and ticks
    plt.grid(True, linestyle="--", alpha=0.7)

    # Add colorbar for the scatter points
    cbar = plt.colorbar(scatter)
    cbar.set_label("Epoch", fontsize=12)

    # Add legend
    plt.legend(fontsize=12)

    # Show epoch markers at reasonable intervals
    max_ticks = 20
    tick_spacing = max(1, len(df) // max_ticks)
    plt.xticks(df["Epoch"][::tick_spacing], fontsize=10)

    plt.tight_layout()

    # Save the plot
    output_path = os.path.join(output_dir, "recurrence_loss_plot.png")
    plt.savefig(output_path, dpi=300)
    print(f"Recurrence loss plot saved to {output_path}")

    return output_path


def plot_training_loss(csv_path, output_dir="evaluation"):
    """
    Creates a simple plot of training loss over epochs with loss axis starting from 0

    Args:
        csv_path: Path to the training loss CSV file
        output_dir: Directory to save the plot
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Read the CSV data
    df = pd.read_csv(csv_path)

    # Set up the plot
    plt.figure(figsize=(10, 6))

    # Simple line plot with markers
    plt.plot(df["Epoch"], df["Loss"], "o-", linewidth=2, markersize=6)

    # Set axis limits and title
    plt.ylim(bottom=0)  # Start loss axis from 0
    plt.title("Model Loss vs Training Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    # Save the plot
    output_path = os.path.join(output_dir, "training_loss_plot.png")
    plt.savefig(output_path)
    print(f"Training loss plot saved to {output_path}")

    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python random_utils.py <command> [args]")
        print("Commands:")
        print("  count_params <checkpoint_path>")
        print("  plot_recurrence <csv_path>")
        print("  plot_training <csv_path>")
        sys.exit(1)

    command = sys.argv[1]

    if command == "count_params" and len(sys.argv) >= 3:
        count_params(sys.argv[2])
    elif command == "plot_recurrence" and len(sys.argv) >= 3:
        csv_path = sys.argv[2]
        output_dir = sys.argv[3] if len(sys.argv) >= 4 else "evaluation"
        plot_recurrence_evaluation(csv_path, output_dir)
    elif command == "plot_training" and len(sys.argv) >= 3:
        csv_path = sys.argv[2]
        output_dir = sys.argv[3] if len(sys.argv) >= 4 else "evaluation"
        plot_training_loss(csv_path, output_dir)
    else:
        print("Invalid command or missing arguments")
        print("Usage: python random_utils.py <command> [args]")
        print("Commands:")
        print("  count_params <checkpoint_path>")
        print("  plot_recurrence <csv_path>")
        print("  plot_training <csv_path>")
