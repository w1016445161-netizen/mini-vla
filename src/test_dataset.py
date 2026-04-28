from pathlib import Path

from torch.utils.data import DataLoader

from src.dataset import MiniVLADataset


def main():
    project_root = Path(__file__).resolve().parent.parent
    dataset_path = project_root / "data" / "processed" / "dataset.json"

    dataset = MiniVLADataset(dataset_path)

    print("数据集总数:", len(dataset))
    print("词表:", dataset.vocab)

    sample = dataset[0]
    print("\n单条样本检查:")
    print("id:", sample["id"])
    print("raw_instruction:", sample["raw_instruction"])
    print("target_color:", sample["target_color"])
    print("image shape:", sample["image"].shape)
    print("instruction tensor:", sample["instruction"])
    print("action tensor:", sample["action"])

    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(loader))

    print("\nBatch 检查:")
    print("image batch shape:", batch["image"].shape)
    print("instruction batch shape:", batch["instruction"].shape)
    print("action batch shape:", batch["action"].shape)
    print("raw instructions:", batch["raw_instruction"])


if __name__ == "__main__":
    main()