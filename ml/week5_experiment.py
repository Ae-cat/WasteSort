"""Controlled Week 5 runs. Preflight is the default; --run explicitly trains."""
import argparse
import collections
import csv
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path


def rows(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preflight(splits=Path('ml/splits_combined.csv'), data_root=Path('data/combined')):
    from PIL import Image
    original, combined = rows('ml/splits.csv'), rows(splits)
    expected = {'train': {'splits.csv': 3323, 'splits_combined.csv': 11426, 'splits_combined_trash.csv': 11563, 'splits_combined_taco.csv': 5314}[Path(splits).name], 'val': 710, 'test': 719}
    if dict(collections.Counter(r['split'] for r in combined)) != expected:
        raise ValueError('Combined split counts differ from John baseline')
    keys = lambda rs, s: [(r['label'], Path(r['path']).name) for r in rs if r['split'] == s]
    for split in ['val', 'test']:
        if keys(original, split) != keys(combined, split):
            raise ValueError(f'{split} membership/order changed')
    identities = [(r['label'], Path(r['path']).name) for r in combined]
    if len(set(identities)) != len(identities):
        raise ValueError('Repeated image identity in combined splits')
    missing = [str(data_root / label / name) for label, name in identities
               if not (data_root / label / name).is_file()]
    if missing:
        raise ValueError(f'Missing {len(missing)} combined images; first: {missing[0]}. Supply Garbage12 and rebuild first.')
    hashes = {}
    for r in combined:
        path = data_root / r['label'] / Path(r['path']).name
        with Image.open(path) as im:
            im.verify()
        h = digest(path)
        if h in hashes and hashes[h] != r['split']:
            raise ValueError(f'Exact image duplicate across splits: {path}')
        hashes[h] = r['split']
    for r in original:
        name = Path(r['path']).name
        if digest(Path('data/realwaste-main/RealWaste') / r['label'] / name) != digest(data_root / r['label'] / name):
            raise ValueError(f'RealWaste image bytes changed: {name}')
    return expected


def configure(model, mode):
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith('fc.') or (mode != 'frozen' and name.startswith('layer4.'))
    return [name for name, p in model.named_parameters() if p.requires_grad]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=Path('reports/week5_controlled_seed42'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mode', choices=['all','frozen','layer4','layer4_aug'], default='all')
    parser.add_argument('--initial-state', type=Path)
    parser.add_argument('--splits', type=Path, default=Path('ml/splits_combined.csv'))
    parser.add_argument('--data-root', type=Path, default=Path('data/combined'))
    args = parser.parse_args()
    counts = preflight(args.splits, args.data_root)
    print('Preflight passed:', counts, flush=True)
    if not args.run:
        return
    if args.output_dir.exists():
        parser.error('Output directory exists; choose a new directory to protect results')
    import numpy as np
    import torch
    import torchvision
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from torchvision.models import ResNet18_Weights
    from dataset import WasteDataset
    from model import build_model
    from train import run_epoch
    from evaluate import evaluate, calculate_metrics, calculate_bin_metrics
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    base_transform = ResNet18_Weights.DEFAULT.transforms()
    if args.initial_state:
        initial_state = torch.load(args.initial_state, map_location='cpu', weights_only=True)
    else:
        initial = build_model(9, pretrained=True, freeze_backbone=True)
        initial_state = {k: v.detach().clone() for k, v in initial.state_dict().items()}
    args.output_dir.mkdir(parents=True)
    config = dict(mode=args.mode, initial_state_source=str(args.initial_state), seed=args.seed, epochs=5, batch_size=32, learning_rate=.001,
                  optimizer='Adam', class_weights=False, device=device,
                  batchnorm='All running statistics update in train mode, matching John train.py',
                  selection='Best validation accuracy; earliest epoch on ties',
                  augmentation=('RandomHorizontalFlip(p=0.5) in layer4_aug only' if args.mode in ['all','layer4_aug'] else 'none'),
                  torch=torch.__version__, torchvision=torchvision.__version__, counts=counts,
                  git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  source_hashes={str(p):digest(p) for p in Path('ml').glob('*.py')},
                  splits=str(args.splits), data_root=str(args.data_root), splits_sha256=digest(args.splits))
    (args.output_dir/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    torch.save(initial_state,args.output_dir/'initial_state.pt')
    results = {}
    modes = ['frozen','layer4','layer4_aug'] if args.mode == 'all' else [args.mode]
    for mode in modes:
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        transform = transforms.Compose([transforms.RandomHorizontalFlip(.5),base_transform]) if mode == 'layer4_aug' else base_transform
        datasets = {s:WasteDataset(s,transform if s=='train' else base_transform,
                                  args.splits,args.data_root) for s in ['train','val','test']}
        generator = torch.Generator().manual_seed(args.seed)
        loaders = {s:DataLoader(ds,batch_size=32,shuffle=s=='train',num_workers=0,
                               generator=generator if s=='train' else None) for s,ds in datasets.items()}
        model = build_model(9,pretrained=False)
        model.load_state_dict(initial_state)
        names = configure(model,mode)
        model.to(device)
        optimizer = torch.optim.Adam(filter(lambda p:p.requires_grad,model.parameters()),lr=.001)
        loss = torch.nn.CrossEntropyLoss()
        run_dir=args.output_dir/mode; run_dir.mkdir()
        (run_dir/'trainable_parameters.json').write_text(json.dumps(names,indent=2)+'\n')
        best=-1.; history=[]
        for epoch in range(1,6):
            start=time.monotonic()
            tl,ta=run_epoch(model,loaders['train'],loss,device,optimizer)
            vl,va=run_epoch(model,loaders['val'],loss,device)
            history.append(dict(epoch=epoch,train_loss=tl,train_accuracy=ta,val_loss=vl,val_accuracy=va,seconds=time.monotonic()-start))
            (run_dir/'history.json').write_text(json.dumps(history,indent=2)+'\n')
            print(mode,history[-1],flush=True)
            if va>best:
                best=va
                torch.save(dict(model_state=model.state_dict(),classes=datasets['train'].classes,epoch=epoch,val_accuracy=va),run_dir/'best.pt')
            # Local recovery snapshot: preserve optimizer and random states at epoch boundaries.
            recovery = dict(model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
                            epoch=epoch, best_accuracy=best, history=history,
                            torch_rng=torch.get_rng_state(), loader_rng=generator.get_state(),
                            python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                            mode=mode, config=config)
            torch.save(recovery, run_dir/'recovery.tmp')
            (run_dir/'recovery.tmp').replace(run_dir/'recovery.pt')
        checkpoint=torch.load(run_dir/'best.pt',map_location=device,weights_only=True)
        model.load_state_dict(checkpoint['model_state'])
        targets,predictions=evaluate(model,loaders['test'],device)
        classes=datasets['test'].classes
        acc,per_class,matrix=calculate_metrics(targets,predictions,classes)
        bin_acc,per_class_bin,bin_matrix=calculate_bin_metrics(matrix,classes)
        result=dict(test_images=len(targets),accuracy=float(acc),bin_accuracy=float(bin_acc),
                    per_class={k:float(v) for k,v in per_class.items()},
                    per_class_bin_accuracy={k:float(v) for k,v in per_class_bin.items()},
                    confusion_matrix=matrix.tolist(),bin_confusion_matrix=bin_matrix.tolist(),
                    classes=classes,best_epoch=checkpoint['epoch'],checkpoint_sha256=digest(run_dir/'best.pt'))
        with (run_dir/'predictions.csv').open('w',newline='') as f:
            writer=csv.writer(f); writer.writerow(['image_path','true_label','predicted_label'])
            writer.writerows((str(sample[0]),classes[t],classes[p]) for sample,t,p in zip(datasets['test'].samples,targets,predictions))
        (run_dir/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
        results[mode]=result
        (args.output_dir/'comparison.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':
    main()
