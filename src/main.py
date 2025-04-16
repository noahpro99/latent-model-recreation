import argparse
from train import train
from test import test

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latent Model Recreation CLI")
    parser.add_argument("mode", choices=["train", "test"], help="Mode to run: train or test")
    parser.add_argument("-c", "--checkpoint", type=str, default=None, help="Path to model checkpoint file")
    parser.add_argument("-e", "--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size for training/testing")
    args = parser.parse_args()

    if args.mode == "train":
        train(checkpoint=args.checkpoint, epochs=args.epochs, batch_size=args.batch_size)
    elif args.mode == "test":
        test(checkpoint=args.checkpoint, batch_size=args.batch_size)
