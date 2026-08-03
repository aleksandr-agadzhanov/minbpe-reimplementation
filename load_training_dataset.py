from datasets import load_dataset

dataset = load_dataset(
    "HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True
)

target_size = 100_000_0
training_dataset_path = "training_datasets/fineweb_edu_1mb.txt"

current_size = 0

with open(training_dataset_path, "w", encoding="utf-8") as f:
    for sample in dataset:
        text = sample["text"]
        f.write(text + "\n")
        current_size += len(text.encode("utf-8"))
        if current_size >= target_size:
            break

print("Finished:", current_size / 1e6, "MB")
