import os

from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()


def load_textbooks_dataset(
    split="train", save_to_file=False, output_path="pretrain_texts.txt"
):

    # Args:
    #    split (str): Dataset split to load (e.g., "train").
    #    save_to_file (bool): If True, saves texts to a .txt file.
    #    output_path (str): Path to save the output text file.

    # Returns:
    #    dataset (datasets.Dataset): Hugging Face Dataset object.

    print(f"Loading '{split}' split")
    dataset = load_dataset(
        "nampdn-ai/tiny-strange-textbooks", split=split, token=os.getenv("HF_TOKEN")
    )
    print(f"Loaded {len(dataset)} examples.")

    if save_to_file:
        print(f"Saving texts to file")
        with open(output_path, "w", encoding="utf-8") as f:
            for example in dataset:
                f.write(example["text"].strip() + "\n")

    return dataset


if __name__ == "__main__":
    SAVE_TO_FILE = True
    OUTPUT_FILE = "pretrain_texts.txt"

    load_textbooks_dataset(save_to_file=SAVE_TO_FILE, output_path=OUTPUT_FILE)
