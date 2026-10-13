"""PyTorch Dataset for the RealWaste train/val/test split defined in ml/splits.csv."""
import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

SPLITS_CSV = Path("ml/splits.csv")


class WasteDataset(Dataset):
    def __init__(self, split, transform=None, csv_path=SPLITS_CSV):
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))

        self.classes = sorted({r["label"] for r in rows})
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples = [(r["path"], self.class_to_idx[r["label"]])
                        for r in rows if r["split"] == split]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
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