import argparse
import datetime
from train import train
from test import manual_test
import os
import glob


def get_latest_checkpoint():
    checkpoint_dir = "./checkpoints"
    if not os.path.isdir(checkpoint_dir):
        return None
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "*"))
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getmtime)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latent Model Recreation CLI")
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Sub-commands")

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        default=f"./checkpoints/checkpoint-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pt",
        help="Path to save the checkpoint file (default: ./checkpoints/checkpoint-<timestamp>.pt)",
    )
    train_parser.add_argument(
        "-e", "--epochs", type=int, default=3, help="Number of training epochs"
    )
    train_parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training",
    )

    # Manual test subcommand
    manual_parser = subparsers.add_parser("manual", help="Run manual test")
    manual_parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        default=get_latest_checkpoint(),
        help="Path to the checkpoint file (default: latest checkpoint in ./checkpoints)",
    )

    args = parser.parse_args()

    if args.mode == "train":
        train(
            checkpoint=args.checkpoint, epochs=args.epochs, batch_size=args.batch_size
        )
    elif args.mode == "manual":
        manual_test(checkpoint=args.checkpoint)
