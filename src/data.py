import os

from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()


def load_textbooks_dataset(
    split="train", save_to_file=False, output_path="pretrain_texts.txt"
):

    print(f"Loading '{split}' split")
    dataset = load_dataset(
        "nampdn-ai/tiny-strange-textbooks",
        split="train[:1000]",
        cache_dir="/mnt/disks/disk-1/cache",
        token=os.getenv("HF_TOKEN"),
    )
    print(f"Loaded {len(dataset)} examples.")
    return dataset


if __name__ == "__main__":
    SAVE_TO_FILE = True
    OUTPUT_FILE = "pretrain_texts.txt"

    load_textbooks_dataset(save_to_file=SAVE_TO_FILE, output_path=OUTPUT_FILE)
