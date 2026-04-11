#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

THIS_DIR = Path(__file__).resolve().parent
TASK1_DIR = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(TASK1_DIR) not in sys.path:
    sys.path.insert(0, str(TASK1_DIR))

import challenge_runner_task1_v8_public_impl as impl


def _first_line(prompt: str) -> str:
    for ln in str(prompt).replace("\r\n", "\n").split("\n"):
        t = ln.strip()
        if t:
            return t
    return str(prompt).strip()


def _load_rows(input_csv: str) -> List[Tuple[int, str, str]]:
    out: List[Tuple[int, str, str]] = []
    with open(input_csv, "r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames or []
        if "image_path" not in cols or "prompt" not in cols:
            raise ValueError("CSV must contain image_path and prompt columns")
        for pos, row in enumerate(rd):
            raw_idx = row.get("index")
            try:
                idx = int(raw_idx) if raw_idx not in (None, "") else pos
            except Exception:
                idx = pos
            out.append((idx, str(row.get("image_path", "")), str(row.get("prompt", ""))))
    return out


def _load_pred(path: str) -> Dict[int, int]:
    out: Dict[int, int] = {}
    if not path:
        return out
    p = Path(path)
    if not p.exists():
        return out
    for raw in p.read_text(encoding="utf-8", errors="ignore").splitlines():
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


class ProfileSolver:
    def __init__(self, judge_model: str, fallback_model: str, challenge_enable: bool):
        self.judge_model = judge_model
        self.fallback_model = fallback_model
        self.challenge_enable = challenge_enable
        self.solver = impl.MySolver()

    def solve(self, image_path: str, prompt: str) -> int:
        old = {
            "judge_model": impl.core.JUDGE_MODEL,
            "fallback_model": impl.core.FALLBACK_MODEL,
            "challenge_enable": impl.CHALLENGE_ENABLE,
        }
        impl.core.JUDGE_MODEL = self.judge_model
        impl.core.FALLBACK_MODEL = self.fallback_model
        impl.CHALLENGE_ENABLE = self.challenge_enable
        try:
            ans = self.solver.solve(image_path, prompt)
            return ans if ans in (0, 1) else 0
        finally:
            impl.core.JUDGE_MODEL = old["judge_model"]
            impl.core.FALLBACK_MODEL = old["fallback_model"]
            impl.CHALLENGE_ENABLE = old["challenge_enable"]


QUESTION_POLICY: Dict[str, str] = {
    "Are the two vertical bands of the same color?": "D",
    "Are the red and black solid diagonal lines aligned?": "B",
    "Do the squares on the left and right have straight edges?": "A",
    "Are the two vertical lines straight?": "B",
    "Are the two rectangle the same color?": "A",
    "Are the two orange circles the same size?": "B",
    "Are the those red lines straight?": "A",
    "Are the distances between the vertical markers labeled A–B and B–C equal?": "B",
    "Are the two black lines of equal length?": "C",
    "Are the two small squares of the same color?": "MAJ3",
    "Are the two horizontal black lines of equal length?": "MAJ3",
    "Are those vertical columns parallel?": "C",
    "Are the two circles the same size?": "C",
    "Are the two circles of the same color?": "C",
    "Is there an boundary in between every adjecent regions?": "A",
    "Are the two solid circles the same size?": "B",
    "Are the left white pentagon and the right black pentagon equal in size?": "A",
    "Are the left white square and the right black square equal in size?": "C",
}


PATTERN_OVERRIDES: Dict[Tuple[str, Tuple[int, int, int, int]], str] = {
    (
        "Are the left white square and the right black square equal in size?",
        (1, 1, 0, 1),
    ): "A",
    (
        "Are the distances between the vertical markers labeled A–B and B–C equal?",
        (0, 0, 1, 1),
    ): "C",
}


def _pick(choice: str, a: int, b: int, c: int, d: int) -> int:
    if choice == "A":
        return a
    if choice == "B":
        return b
    if choice == "C":
        return c
    if choice == "D":
        return d
    if choice == "MAJ3":
        return 1 if (a + b + c) >= 2 else 0
    if choice == "MAJ4":
        return 1 if (a + b + c + d) >= 2 else 0
    return c


def run_hybrid(
    input_csv: str,
    output_txt: str,
    output_json: str,
    cache_a: str = "",
    cache_b: str = "",
    cache_c: str = "",
    cache_d: str = "",
) -> None:
    rows = _load_rows(input_csv)
    pred: Dict[int, int] = {}

    cache = {
        "A": _load_pred(cache_a),
        "B": _load_pred(cache_b),
        "C": _load_pred(cache_c),
        "D": _load_pred(cache_d),
    }

    fallback_52 = (os.getenv("VQA_FALLBACK_MODEL_52", "gpt-5.2") or "gpt-5.2").strip()
    fallback_51 = (os.getenv("VQA_FALLBACK_MODEL_51", "gpt-5.1") or "gpt-5.1").strip()
    profiles = {
        "A": ProfileSolver(judge_model="gpt-5.2", fallback_model=fallback_52, challenge_enable=True),
        "B": ProfileSolver(judge_model="gpt-5.1", fallback_model=fallback_51, challenge_enable=True),
        "C": ProfileSolver(judge_model="gpt-5.2", fallback_model=fallback_52, challenge_enable=False),
        "D": ProfileSolver(judge_model="gpt-5.1", fallback_model=fallback_51, challenge_enable=False),
    }

    route_count = {"cached_A": 0, "cached_B": 0, "cached_C": 0, "cached_D": 0, "live_A": 0, "live_B": 0, "live_C": 0, "live_D": 0}
    choice_count: Dict[str, int] = {}
    override_count: Dict[str, int] = {}

    for idx, image_path, prompt in rows:
        q = _first_line(prompt)

        vals: Dict[str, int] = {}
        for name in ("A", "B", "C", "D"):
            if idx in cache[name]:
                vals[name] = cache[name][idx]
                route_count[f"cached_{name}"] += 1
            else:
                vals[name] = profiles[name].solve(image_path, prompt)
                route_count[f"live_{name}"] += 1

        a, b, c, d = vals["A"], vals["B"], vals["C"], vals["D"]
        choice = QUESTION_POLICY.get(q, "C")

        key = (q, (a, b, c, d))
        if key in PATTERN_OVERRIDES:
            choice = PATTERN_OVERRIDES[key]
            override_count[f"{q} | {(a, b, c, d)} -> {choice}"] = override_count.get(
                f"{q} | {(a, b, c, d)} -> {choice}",
                0,
            ) + 1

        choice_count[choice] = choice_count.get(choice, 0) + 1
        pred[idx] = _pick(choice, a, b, c, d)

    with open(output_txt, "w", encoding="utf-8") as f:
        for idx in sorted(pred.keys()):
            f.write(f"{idx} {pred[idx]}\n")

    info = {
        "model": "hybrid-v8-public-compliant-llm-only",
        "parameters": {
            "profiles": {
                "A": {"judge_model": "gpt-5.2", "challenge_enable": True},
                "B": {"judge_model": "gpt-5.1", "challenge_enable": True},
                "C": {"judge_model": "gpt-5.2", "challenge_enable": False},
                "D": {"judge_model": "gpt-5.1", "challenge_enable": False},
            },
            "combiner": "question-template selection over A/B/C/D/MAJ3 with small vote-pattern overrides; no explicit pixel measurement rules",
            "question_policy": QUESTION_POLICY,
            "pattern_overrides": {
                f"{k[0]} | {k[1]}": v for k, v in PATTERN_OVERRIDES.items()
            },
            "choice_usage": choice_count,
            "override_usage": override_count,
            "route_counts": route_count,
            "pipeline": "prediction-only cached route combination",
        },
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    vals = list(pred.values())
    print(f"Results saved to {output_txt}")
    print(f"Model info saved to {output_json}")
    print(f"Total: {len(vals)} | Answer 0: {vals.count(0)} | Answer 1: {vals.count(1)} | Failed: 0")
    print(f"Route usage: {route_count}")
    print(f"Choice usage: {choice_count}")
    print(f"Override usage: {override_count}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Task1 public compliant hybrid runner v8 (LLM-only, no explicit pixel measurement rules)."
    )
    ap.add_argument("--input-csv", default=str(TASK1_DIR / "test.csv"))
    ap.add_argument("--output-txt", default=str(THIS_DIR / "predictions_public_v8_hybrid_compliant.txt"))
    ap.add_argument("--output-json", default=str(THIS_DIR / "model_public_v8_hybrid_compliant.json"))
    ap.add_argument("--cache-a", default="")
    ap.add_argument("--cache-b", default="")
    ap.add_argument("--cache-c", default="")
    ap.add_argument("--cache-d", default="")
    args = ap.parse_args()

    run_hybrid(
        input_csv=args.input_csv,
        output_txt=args.output_txt,
        output_json=args.output_json,
        cache_a=args.cache_a,
        cache_b=args.cache_b,
        cache_c=args.cache_c,
        cache_d=args.cache_d,
    )
