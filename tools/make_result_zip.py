#!/usr/bin/env python3
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description='Create submission zip from prediction.txt and model.json.')
    ap.add_argument('--prediction', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    prediction = Path(args.prediction)
    model = Path(args.model)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(prediction, arcname='prediction.txt')
        zf.write(model, arcname='model.json')


if __name__ == '__main__':
    main()
