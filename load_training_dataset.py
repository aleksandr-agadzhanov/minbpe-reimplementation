from datasets import load_dataset

dataset = load_dataset(
    "HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True
)

TARGET_SIZE_BYTES = 100_000_000
TRAINING_DATASET_PATH = "training_datasets/fineweb_edu_100mb.txt"

current_size = 0

with open(TRAINING_DATASET_PATH, "w", encoding="utf-8") as f:
    for sample in dataset:
        text = sample["text"]
        f.write(text + "\n")
        current_size += len(text.encode("utf-8"))
        if current_size >= TARGET_SIZE_BYTES:
            break

print("Finished:", current_size / 1e6, "MB")
