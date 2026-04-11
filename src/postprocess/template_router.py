#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple


THIS_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = THIS_DIR.parent.parent
POLICY_JSON_DEFAULT = PACKAGE_ROOT / "policies" / "template_router_policy_v2.json"

ROUTE_ORDER: Tuple[str, ...] = ("A", "B", "C", "D", "L", "W", "PUB", "RR2", "TREE", "V9")
MAJ_KEYS: Tuple[str, ...] = ("A", "B", "C", "D", "L", "W", "PUB", "RR2")


def _display_path(path: str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(PACKAGE_ROOT))
    except Exception:
        pass
    parts = p.parts
    if len(parts) >= 3:
        return str(Path(*parts[-3:]))
    return p.name or str(path)


def _first_line(prompt: str) -> str:
    for line in str(prompt).replace("\r\n", "\n").split("\n"):
        text = line.strip()
        if text:
            return text
    return str(prompt).strip()


def _load_rows(input_csv: str) -> List[Tuple[int, str]]:
    rows: List[Tuple[int, str]] = []
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
            rows.append((idx, str(row.get("prompt", ""))))
    return rows


def _load_pred(path: str) -> Dict[int, int]:
    out: Dict[int, int] = {}
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


def _pick(action: str, vals: Dict[str, int]) -> int:
    if action == "MAJ":
        votes = [vals.get(k, 0) for k in MAJ_KEYS]
        return 1 if sum(votes) >= 5 else 0
    if action.startswith("MAJ3:"):
        keys = [k.strip() for k in action.split(":", 1)[1].split("+") if k.strip()]
        if len(keys) != 3:
            raise ValueError(f"invalid MAJ3 action: {action}")
        return 1 if sum(vals.get(k, 0) for k in keys) >= 2 else 0
    return 1 if vals.get(action, 0) == 1 else 0


def _load_policy(policy_json: str) -> Tuple[Dict[str, str], dict]:
    obj = json.loads(Path(policy_json).read_text(encoding="utf-8"))
    if isinstance(obj.get("prompt_policy"), dict):
        policy = {str(q): str(action) for q, action in obj["prompt_policy"].items()}
    else:
        raw = obj.get("per_prompt_best_action", {})
        policy = {str(q): str(meta.get("best_action", "RR2")) for q, meta in raw.items()}
    return policy, obj


def run_router(
    input_csv: str,
    output_txt: str,
    output_json: str,
    profile_paths: Dict[str, str],
    policy_json: str,
    default_action: str,
) -> None:
    policy, meta = _load_policy(policy_json)
    rows = _load_rows(input_csv)
    pred_maps = {k: _load_pred(v) for k, v in profile_paths.items()}
    missing = [k for k, v in pred_maps.items() if not v]
    if missing:
        print(f"[WARN] empty/missing profile predictions: {missing}")

    pred: Dict[int, int] = {}
    action_usage: Dict[str, int] = {}
    question_usage: Dict[str, Dict[str, int]] = {}

    for idx, prompt in rows:
        q = _first_line(prompt)
        vals = {k: (1 if pred_maps.get(k, {}).get(idx, 0) == 1 else 0) for k in ROUTE_ORDER}
        action = policy.get(q, default_action)
        pred[idx] = _pick(action, vals)
        action_usage[action] = action_usage.get(action, 0) + 1
        if q not in question_usage:
            question_usage[q] = {"count": 0, "action": action}
        question_usage[q]["count"] += 1

    with open(output_txt, "w", encoding="utf-8") as f:
        for idx in sorted(pred.keys()):
            f.write(f"{idx} {pred[idx]}\n")

    info = {
        "model": "template-router-v2",
        "parameters": {
            "profile_paths": {k: _display_path(v) for k, v in profile_paths.items()},
            "route_order": ROUTE_ORDER,
            "majority_keys": MAJ_KEYS,
            "default_action": default_action,
            "policy_json": _display_path(policy_json),
            "prompt_policy": policy,
            "question_usage": question_usage,
            "action_usage": action_usage,
            "policy_metadata": {
                "kind": meta.get("kind"),
                "description": meta.get("description"),
            },
            "pipeline": "template router over cached upstream route outputs",
            "no_minus_one": True,
        },
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    vals = list(pred.values())
    print(f"Results saved to {_display_path(output_txt)}")
    print(f"Model info saved to {_display_path(output_json)}")
    print(f"Total: {len(vals)} | Answer 0: {vals.count(0)} | Answer 1: {vals.count(1)} | Failed: 0")
    print(f"Action usage: {action_usage}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Template router over cached route outputs.")
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--output-txt", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--policy-json", default=str(POLICY_JSON_DEFAULT))
    ap.add_argument("--default-action", default="RR2")
    ap.add_argument("--cache-a", required=True)
    ap.add_argument("--cache-b", required=True)
    ap.add_argument("--cache-c", required=True)
    ap.add_argument("--cache-d", required=True)
    ap.add_argument("--cache-l", required=True)
    ap.add_argument("--cache-w", required=True)
    ap.add_argument("--cache-pub", required=True)
    ap.add_argument("--cache-rr2", required=True)
    ap.add_argument("--cache-tree", required=True)
    ap.add_argument("--cache-v9", required=True)
    args = ap.parse_args()

    profile_paths = {
        "A": args.cache_a,
        "B": args.cache_b,
        "C": args.cache_c,
        "D": args.cache_d,
        "L": args.cache_l,
        "W": args.cache_w,
        "PUB": args.cache_pub,
        "RR2": args.cache_rr2,
        "TREE": args.cache_tree,
        "V9": args.cache_v9,
    }
    run_router(
        input_csv=args.input_csv,
        output_txt=args.output_txt,
        output_json=args.output_json,
        profile_paths=profile_paths,
        policy_json=args.policy_json,
        default_action=args.default_action,
    )


if __name__ == "__main__":
    main()
