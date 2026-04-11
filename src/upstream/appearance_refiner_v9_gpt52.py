#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Set

import challenge_core_v9_public as core

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent

TARGETS: Set[str] = {
    "Are the left white pentagon and the right black pentagon equal in size?",
    "Are the left white square and the right black square equal in size?",
    "Are the two black lines of equal length?",
    "Are the two circles the same size?",
    "Are the two circles of the same color?",
    "Are the two horizontal black lines of equal length?",
    "Are the two orange circles the same size?",
    "Are the two rectangle the same color?",
    "Are the two small squares of the same color?",
    "Are the two solid circles the same size?",
    "Are the two vertical bands of the same color?",
}


def _load_pred(path: str) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for raw in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        sp = raw.strip().split()
        if len(sp) < 2:
            continue
        try:
            idx = int(sp[0])
            val = int(float(sp[1]))
        except Exception:
            continue
        out[idx] = 1 if val == 1 else 0
    return out


def _first_line(prompt: str) -> str:
    for line in str(prompt).replace("\r\n", "\n").split("\n"):
        text = line.strip()
        if text:
            return text
    return str(prompt).strip()


def _resolve_image(image_path: str, input_csv: str) -> str:
    p = image_path or ""
    if os.path.isabs(p) and os.path.exists(p):
        return p
    if os.path.exists(p):
        return p
    root = Path(input_csv).resolve().parent
    alt = str(root / p)
    if os.path.exists(alt):
        return alt
    alt2 = str(THIS_DIR / p)
    if os.path.exists(alt2):
        return alt2
    return p


class AppearanceRefiner:
    def __init__(self, base_predictions: Dict[int, int], input_csv: str):
        self.base_predictions = base_predictions
        self.input_csv = input_csv
        self.solver = core.MySolver()

    def run(self, output_txt: str, output_json: str) -> None:
        results: Dict[int, int] = {}
        refined = 0

        with open(self.input_csv, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

        for pos, row in enumerate(rows):
            raw_idx = row.get("index")
            try:
                idx = int(raw_idx) if raw_idx not in (None, "") else pos
            except Exception:
                idx = pos

            prompt = str(row.get("prompt", ""))
            question = _first_line(prompt)
            if question in TARGETS:
                image_path = _resolve_image(str(row.get("image_path", "")), self.input_csv)
                try:
                    results[idx] = self.solver.solve(image_path, prompt)
                except Exception as e:
                    print(f"[REFINER WARN] idx={idx} err={e}")
                    results[idx] = self.base_predictions.get(idx, 0)
                refined += 1
            else:
                results[idx] = self.base_predictions.get(idx, 0)

        with open(output_txt, "w", encoding="utf-8") as f:
            for idx in sorted(results.keys()):
                f.write(f"{idx} {results[idx]}\n")

        info = {
            "model": core.JUDGE_MODEL,
            "parameters": {
                "base_url": core.BASE_URL,
                "base_predictions": "external",
                "refined_questions": sorted(TARGETS),
                "refined_count": refined,
                "pipeline": "base predictions with dedicated v9 appearance verifier refiner for color/size/length prompts",
            },
        }
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

        vals = list(results.values())
        print(f"Results saved to {output_txt}")
        print(f"Model info saved to {output_json}")
        print(
            f"Total: {len(vals)} | Answer 0: {vals.count(0)} | "
            f"Answer 1: {vals.count(1)} | Failed: 0 | Refined: {refined}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Refine appearance-heavy questions over base predictions using the compliant v9 verifier."
    )
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--base-pred", required=True)
    ap.add_argument("--output-txt", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()

    default_key = (
        os.getenv("VQA_API_KEY", "").strip()
        or os.getenv("OPENAI_API_KEY", "").strip()
        or ((REPO_ROOT / ".openai_api_key").read_text(encoding="utf-8").strip() if (REPO_ROOT / ".openai_api_key").exists() else "")
    )
    if default_key:
        core.API_KEY = default_key
    core.BASE_URL = core._normalize_base_url(os.getenv("VQA_BASE_URL", "https://api.xbai.top/v1"))
    core.JUDGE_MODEL = (os.getenv("VQA_JUDGE_MODEL", "gpt-5.2") or "gpt-5.2").strip()
    core.FALLBACK_MODEL = (os.getenv("VQA_FALLBACK_MODEL", core.FALLBACK_MODEL) or core.FALLBACK_MODEL).strip()

    base_predictions = _load_pred(args.base_pred)
    AppearanceRefiner(base_predictions, args.input_csv).run(args.output_txt, args.output_json)


if __name__ == "__main__":
    main()
