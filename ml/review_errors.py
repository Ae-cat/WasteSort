"""Build a sortable-by-confidence error inventory and local HTML contact sheet."""
import argparse
import csv
import html
import json
import os
from collections import Counter
from pathlib import Path
from urllib.parse import quote

from PIL import Image, ImageDraw, ImageFont, ImageOps

from model import DISPOSAL_MAP


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, default=Path('reports/week3_held_out_test/predictions.csv'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Output already exists; choose a new directory')
    with args.predictions.open(newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    if not rows or any(r['dataset_kind'] != 'realwaste_held_out_test' for r in rows):
        raise ValueError('Expected labeled held-out RealWaste predictions')
    errors = sorted([r for r in rows if r['true_label'] != r['predicted_label']],
                    key=lambda r: float(r['confidence']), reverse=True)
    for row in errors:
        row['disposal_correct'] = DISPOSAL_MAP[row['true_label']] == DISPOSAL_MAP[row['predicted_label']]
        row['review_note'] = ''
    pairs = Counter((r['true_label'], r['predicted_label']) for r in errors)
    supports = Counter(r['true_label'] for r in rows)
    counts = Counter(r['true_label'] for r in errors)
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / 'errors.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields + ['disposal_correct', 'review_note'])
        writer.writeheader()
        writer.writerows(errors)
    summary = dict(total=len(rows), errors=len(errors), disposal_errors=sum(not r['disposal_correct'] for r in errors),
                   confidence_at_least_90pct_errors=sum(float(r['confidence']) >= .9 for r in errors),
                   by_class={c: dict(support=supports[c], errors=counts[c]) for c in sorted(supports)},
                   confusion_pairs=[dict(true_label=a, predicted_label=b, count=n)
                                    for (a,b),n in pairs.most_common()])
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    cards = []
    for i, row in enumerate(errors, 1):
        path = Path(row['image_path'])
        if not path.is_file():
            raise FileNotFoundError(path)
        url = quote(os.path.relpath(path.resolve(), args.output_dir.resolve()))
        label = f"{row['true_label']} → {row['predicted_label']}"
        cards.append(f'''<article><a href="{url}"><img loading="lazy" src="{url}" alt="{html.escape(label)}"></a>
<h2>#{i} {html.escape(label)}</h2><p>Confidence {float(row['confidence']):.1%} ·
Disposal {'correct' if row['disposal_correct'] else 'WRONG'}</p>
<p>{html.escape(str(path))}</p></article>''')
    top = errors[:12]
    if top:
        sheet = Image.new('RGB', (1200, ((len(top) + 3) // 4) * 345), 'white')
        draw = ImageDraw.Draw(sheet)
        font = ImageFont.load_default(size=15)
        for i, row in enumerate(top):
            x, y = (i % 4) * 300, (i // 4) * 345
            with Image.open(row['image_path']) as image:
                thumb = ImageOps.contain(image.convert('RGB'), (290, 260))
                sheet.paste(thumb, (x + (300 - thumb.width) // 2, y))
            lines = [f"#{i+1} True: {row['true_label']}", f"Pred: {row['predicted_label']}",
                     f"Confidence {float(row['confidence']):.1%}"]
            for j, line in enumerate(lines):
                draw.text((x + 6, y + 264 + 21 * j), line, fill='black', font=font)
        sheet.save(args.output_dir / 'top12_errors.jpg')
    document = '''<!doctype html><html lang="en"><meta charset="utf-8">
<title>WasteSort baseline error review</title><style>
body{font:16px system-ui;margin:30px;background:#f5f5f5;color:#162331}
main{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:18px}
article{background:white;padding:14px;border:1px solid #ddd;border-radius:10px}
img{width:100%;height:240px;object-fit:contain}h2{font-size:17px}p{overflow-wrap:anywhere}
</style><h1>Baseline held-out RealWaste error review</h1>
<p>Sorted by decreasing confidence. These are not external images. True labels are dataset labels;
visual cause must be reviewed, not inferred from the prediction. CSV contains an empty review_note column.</p><main>'''
    (args.output_dir / 'index.html').write_text(document + '\n'.join(cards) + '</main></html>')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
