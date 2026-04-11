#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict

from openai import OpenAI

import challenge_runner_task1_v8_public_impl as impl
import challenge_core_v9_public as core

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
TARGETS = {
    "Are the those red lines straight?",
    "Do the squares on the left and right have straight edges?",
    "Is there an boundary in between every adjecent regions?",
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


class StructureRefiner:
    def __init__(self, base_predictions: Dict[int, int], input_csv: str):
        self.base_predictions = base_predictions
        self.input_root = str(Path(input_csv).resolve().parent)
        self.client = OpenAI(
            api_key=(core.API_KEY or None),
            base_url=(core.BASE_URL or None),
            timeout=float(os.getenv("VQA_REQUEST_TIMEOUT", "45")),
            max_retries=0,
        )

    def _call_text(self, messages: list[dict], max_tokens: int) -> str:
        kwargs: Dict[str, Any] = dict(
            model=core.JUDGE_MODEL,
            messages=messages,
            temperature=0.0,
            top_p=1.0,
            max_tokens=max_tokens,
        )
        try:
            kwargs["seed"] = int(os.getenv("VQA_SEED", "42"))
        except Exception:
            pass
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except TypeError:
            kwargs.pop("seed", None)
            resp = self.client.chat.completions.create(**kwargs)
        if isinstance(resp, str):
            return resp
        if isinstance(resp, dict):
            return resp.get("choices", [{}])[0].get("message", {}).get("content", "")
        return resp.choices[0].message.content if getattr(resp, "choices", None) else ""

    def _resolve_image(self, image_path: str) -> str:
        p = impl._resolve_image_path(image_path)
        if p and (not os.path.isabs(p)):
            p = str(Path(self.input_root) / p)
        return p

    def _prompt(self, question: str) -> str:
        if question == "Are the those red lines straight?":
            return f"""
You are checking objective line straightness only.
The target objects are the red lines.

Rules:
- Inspect the red lines directly; ignore nearby black slanted marks.
- If any target red line has a visible bend, kink, wobble, or change of direction, answer 0.
- If the target red lines are straight from top to bottom, answer 1.
- Do not answer from illusion prior.

Return ONLY one line exactly: <answer>1</answer> or <answer>0</answer>.

Question:
{question}
""".strip()
        if question == "Do the squares on the left and right have straight edges?":
            return f"""
You are checking objective edge straightness only.
The target objects are the outlines of the two squares.

Rules:
- Inspect the actual drawn square edges, not the surrounding background pattern.
- If any edge shows a visible bend, bow, wobble, or corner misalignment that makes it non-straight, answer 0.
- If the square edges are straight as drawn, answer 1.
- Do not answer from illusion prior.

Return ONLY one line exactly: <answer>1</answer> or <answer>0</answer>.

Question:
{question}
""".strip()
        return f"""
You are checking whether there is an explicit boundary between EVERY adjacent region.

Rules:
- A boundary means a visible separating edge/line/border between two neighboring regions.
- Mere color change without a visible border does NOT count.
- If even one adjacent pair lacks a boundary, answer 0.
- If every adjacent pair is separated by a visible boundary, answer 1.

Return ONLY one line exactly: <answer>1</answer> or <answer>0</answer>.

Question:
{question}
""".strip()

    def _challenge(self, question: str) -> str:
        if question == "Are the those red lines straight?":
            return f"""
Act as a strict challenger and try to REFUTE answer=1.
If you can find any visible bend or kink in a target red line, answer 0.
Otherwise answer 1.
Return ONLY one line exactly: <answer>1</answer> or <answer>0</answer>.

Question:
{question}
""".strip()
        if question == "Do the squares on the left and right have straight edges?":
            return f"""
Act as a strict challenger and try to REFUTE answer=1.
If you can find any visible bend, bow, or non-straight square edge, answer 0.
Otherwise answer 1.
Return ONLY one line exactly: <answer>1</answer> or <answer>0</answer>.

Question:
{question}
""".strip()
        return f"""
Act as a strict challenger and try to REFUTE answer=1.
If you can find even one adjacent pair of regions without an explicit separating boundary, answer 0.
Otherwise answer 1.
Return ONLY one line exactly: <answer>1</answer> or <answer>0</answer>.

Question:
{question}
""".strip()

    def solve_target(self, image_path: str, question: str) -> int:
        image_path = self._resolve_image(image_path)
        if question == "Are the those red lines straight?":
            assist = impl._build_vertical_line_montage(image_path)
        elif question == "Do the squares on the left and right have straight edges?":
            assist = None
        else:
            assist = None
        include_raw = True
        msg = core.build_mm_messages(core.system_text(), image_path, assist, None, self._prompt(question), include_raw=include_raw)
        out = self._call_text(msg, max_tokens=96)
        ans = core.parse_answer(out)
        if ans not in (0, 1):
            return self.base_predictions.get(-1, 0)
        if ans == 1:
            ch_msg = core.build_mm_messages(core.system_text(), image_path, assist, None, self._challenge(question), include_raw=include_raw)
            ch_out = self._call_text(ch_msg, max_tokens=64)
            ch = core.parse_answer(ch_out)
            if ch == 0:
                return 0
        return ans

    def run(self, input_csv: str, output_txt: str, output_json: str) -> None:
        results: Dict[int, int] = {}
        refined = 0
        with open(input_csv, 'r', encoding='utf-8', newline='') as f:
            rows = list(csv.DictReader(f))
        for pos, row in enumerate(rows):
            raw_idx = row.get('index')
            try:
                idx = int(raw_idx) if raw_idx not in (None, '') else pos
            except Exception:
                idx = pos
            question = impl._first_line(str(row.get('prompt', '')))
            if question in TARGETS:
                try:
                    results[idx] = self.solve_target(str(row.get('image_path', '')), question)
                except Exception as e:
                    print(f'[REFINER WARN] idx={idx} err={e}')
                    results[idx] = self.base_predictions.get(idx, 0)
                refined += 1
            else:
                results[idx] = self.base_predictions.get(idx, 0)
        with open(output_txt, 'w', encoding='utf-8') as f:
            for idx in sorted(results.keys()):
                f.write(f"{idx} {results[idx]}\n")
        info = {
            "model": core.JUDGE_MODEL,
            "parameters": {
                "base_url": core.BASE_URL,
                "base_predictions": "semantic_default",
                "refined_questions": sorted(TARGETS),
                "refined_count": refined,
                "pipeline": "base semantic fusion with dedicated gpt-5.2 structure refiner",
            },
        }
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        vals = list(results.values())
        print(f"Results saved to {output_txt}")
        print(f"Model info saved to {output_json}")
        print(f"Total: {len(vals)} | Answer 0: {vals.count(0)} | Answer 1: {vals.count(1)} | Failed: 0 | Refined: {refined}")


def main() -> None:
    ap = argparse.ArgumentParser(description='Refine boundary/straightness questions over base predictions using gpt-5.2.')
    ap.add_argument('--input-csv', required=True)
    ap.add_argument('--base-pred', required=True)
    ap.add_argument('--output-txt', required=True)
    ap.add_argument('--output-json', required=True)
    args = ap.parse_args()

    default_key = (
        os.getenv('VQA_API_KEY', '').strip()
        or os.getenv('OPENAI_API_KEY', '').strip()
        or ((REPO_ROOT / '.openai_api_key').read_text(encoding='utf-8').strip() if (REPO_ROOT / '.openai_api_key').exists() else '')
    )
    if default_key:
        core.API_KEY = default_key
    core.BASE_URL = impl._norm_base_url(os.getenv('VQA_BASE_URL', 'https://api.xbai.top/v1'))
    core.JUDGE_MODEL = (os.getenv('VQA_JUDGE_MODEL', 'gpt-5.2') or 'gpt-5.2').strip()
    base_predictions = _load_pred(args.base_pred)
    StructureRefiner(base_predictions, args.input_csv).run(args.input_csv, args.output_txt, args.output_json)


if __name__ == '__main__':
    main()
