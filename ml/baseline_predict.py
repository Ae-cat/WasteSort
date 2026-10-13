from pathlib import Path
import random
import torch
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights

ROOT = Path("data/realwaste-main/RealWaste")
random.seed(0)

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.eval()
preprocess = weights.transforms()          
labels = weights.meta["categories"]       

for d in sorted(p for p in ROOT.iterdir() if p.is_dir()):
    f = random.choice(list(d.glob("*.jpg")))
    x = preprocess(Image.open(f)).unsqueeze(0)  
    with torch.no_grad():
        probs = model(x).softmax(dim=1)[0]
    conf, idx = probs.max(dim=0)
    print(f"{d.name:20s} -> {labels[idx]} ({conf.item():.0%})")