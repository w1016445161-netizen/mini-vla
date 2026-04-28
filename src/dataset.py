import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MiniVLADataset(Dataset):
    def __init__(self, json_path, vocab=None, max_text_len=6):
        self.project_root = Path(__file__).resolve().parent.parent
        self.max_text_len = max_text_len

        with open(json_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
        ])

        if vocab is None:
            self.vocab = self.build_vocab(self.samples)
        else:
            self.vocab = vocab

    def build_vocab(self, samples):
        words = set()
        for sample in samples:
            for word in sample["instruction"].lower().split():
                words.add(word)

        vocab = {"<pad>": 0, "<unk>": 1}
        for i, word in enumerate(sorted(words), start=2):
            vocab[word] = i

        return vocab

    def encode_instruction(self, text):
        tokens = []
        for word in text.lower().split():
            tokens.append(self.vocab.get(word, self.vocab["<unk>"]))

        if len(tokens) < self.max_text_len:
            tokens += [self.vocab["<pad>"]] * (self.max_text_len - len(tokens))
        else:
            tokens = tokens[:self.max_text_len]

        return torch.tensor(tokens, dtype=torch.long)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image_path = self.project_root / sample["image"]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        instruction = self.encode_instruction(sample["instruction"])
        action = torch.tensor(sample["action"], dtype=torch.float32)

        agent_pos = torch.tensor(sample["agent_pos"], dtype=torch.float32)

        blocks = {
            color: torch.tensor(pos, dtype=torch.float32)
            for color, pos in sample["blocks"].items()
        }

        return {
            "image": image,
            "instruction": instruction,
            "action": action,
            "raw_instruction": sample["instruction"],
            "target_color": sample["target_color"],
            "id": sample["id"],
            "agent_pos": agent_pos,
            "blocks": blocks,
        }