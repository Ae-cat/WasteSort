"""PyTorch Dataset for the RealWaste train/val/test split."""
import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

SPLITS_CSV = Path("ml/splits.csv")


class WasteDataset(Dataset):
    def __init__(self, split, transform=None, csv_path=SPLITS_CSV, data_root=None):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be 'train', 'val', or 'test'")
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))

        self.classes = sorted({r["label"] for r in rows})
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.data_root = Path(data_root) if data_root else None
        self.samples = [
            (self._resolve_path(r), self.class_to_idx[r["label"]])
            for r in rows if r["split"] == split
        ]
        self.transform = transform

        if not self.samples:
            raise ValueError(f"No samples found for split '{split}' in {csv_path}")

    def _resolve_path(self, row):
        if self.data_root is not None:
            return self.data_root / row["label"] / Path(row["path"]).name
        return Path(row["path"])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        with Image.open(path) as source:
            img = source.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


if __name__ == "__main__":

    from torch.utils.data import DataLoader
    from torchvision.models import ResNet18_Weights

    tf = ResNet18_Weights.DEFAULT.transforms()
    for split in ["train", "val", "test"]:
        ds = WasteDataset(split, transform=tf)
        print(f"{split}: {len(ds)} images")

    loader = DataLoader(WasteDataset("train", transform=tf), batch_size=8, shuffle=True)
    x, y = next(iter(loader))
    print("batch shape:", x.shape)      
    print("labels:", y.tolist())
    print("classes:", WasteDataset("train").classes)
