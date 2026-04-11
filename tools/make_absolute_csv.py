#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description='Convert image_path in csv to absolute paths.')
    ap.add_argument('--input', required=True)
    ap.add_argument('--task1-root', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    task1_root = Path(args.task1_root).resolve()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.input, 'r', encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
        cols = rows[0].keys() if rows else ['index', 'image_path', 'prompt']

    with open(out_path, 'w', encoding='utf-8', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(cols))
        wr.writeheader()
        for row in rows:
            image_path = (row.get('image_path') or '').strip()
            row['image_path'] = str((task1_root / image_path).resolve()) if image_path else image_path
            wr.writerow(row)


if __name__ == '__main__':
    main()
