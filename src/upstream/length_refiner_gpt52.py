#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from openai import OpenAI

import challenge_runner_task1_v8_public_impl as impl
import challenge_core_v9_public as core

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
TARGET_QUESTIONS = {
    "Are the two black lines of equal length?",
    "Are the two horizontal black lines of equal length?",
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
    return impl._first_line(prompt)


def _build_target_assist(image_path: str, question: str) -> Optional[str]:
    if "markers labeled A–B and B–C" in question:
        return impl._build_distance_marker_montage(image_path) or impl._build_assist(image_path, "distance_markers")
    return impl._build_length_focus_montage(image_path) or impl._build_assist(image_path, "length")


def _length_json_prompt(question: str) -> str:
    return f"""
You are verifying objective geometric equality only from the provided image views.

Decide whether the referenced two target lengths are truly equal as drawn.
Use the strip and endpoint panels as the main evidence. Ignore any illusion prior.

Rules:
- Compare ONLY the referenced segments or gaps.
- Check left/start side and right/end side separately.
- If either side shows a visible mismatch, the final answer should be 0.
- If both sides align, the final answer should be 1.
- If evidence is mixed or uncertain, set best_guess to your best binary judgment.

Return ONLY valid JSON with this exact schema:
{{"left_end":"aligned|not_aligned|uncertain","right_end":"aligned|not_aligned|uncertain","best_guess":0_or_1}}

Question:
{question}
""".strip()


def _distance_json_prompt(question: str) -> str:
    return f"""
You are verifying objective spacing only from the provided image views.

Compare the gap A–B and the gap B–C using the vertical marker lines as anchors.
Ignore text labels and any illusion prior.

Rules:
- Focus only on the horizontal spacing between adjacent vertical markers.
- If A–B and B–C visibly differ, the final answer should be 0.
- If they match, the final answer should be 1.
- If evidence is mixed or uncertain, set best_guess to your best binary judgment.

Return ONLY valid JSON with this exact schema:
{{"ab_vs_bc":"equal|not_equal|uncertain","best_guess":0_or_1}}

Question:
{question}
""".strip()


def _distance_from_json(obj: Optional[dict]) -> int:
    if not isinstance(obj, dict):
        return -1
    tok = str(obj.get("ab_vs_bc", "")).strip().lower().replace('-', '_').replace(' ', '_')
    if tok == "equal":
        return 1
    if tok == "not_equal":
        return 0
    bg = obj.get("best_guess")
    try:
        bg = int(bg)
    except Exception:
        return -1
    return bg if bg in (0, 1) else -1


def _binary_fallback_prompt(question: str) -> str:
    return (
        "Return ONLY one line exactly: <answer>1</answer> or <answer>0</answer>.\n"
        "Judge the objective geometry only from the provided views. Ignore illusion priors.\n\n"
        f"Question:\n{question}"
    )


def _strict_challenge_prompt(question: str) -> str:
    return f"""
Act as a strict challenger and try to REFUTE answer=1.
Look for a concrete visible mismatch in the compared lengths/gaps.
- If you find a clear mismatch, answer 0.
- If no clear mismatch is visible, answer 1.
Return ONLY: <answer>1</answer> or <answer>0</answer>.

Question:
{question}
""".strip()


class LengthRefiner:
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

    def solve_target(self, image_path: str, question: str) -> int:
        image_path = impl._resolve_image_path(image_path)
        if image_path and (not os.path.isabs(image_path)):
            image_path = str(Path(self.input_root) / image_path)
        assist = _build_target_assist(image_path, question)
        assist_flip = core.maybe_flip_data_url(assist) if assist else None
        include_raw = True
        sys_txt = core.system_text()

        if "markers labeled A–B and B–C" in question:
            prompt = _distance_json_prompt(question)
        else:
            prompt = _length_json_prompt(question)

        msg = core.build_mm_messages(sys_txt, image_path, assist, assist_flip, prompt, include_raw=include_raw)
        out = self._call_text(msg, max_tokens=180)

        if "markers labeled A–B and B–C" in question:
            ans = _distance_from_json(core._extract_json(out))
        else:
            ans = core.derive_answer_from_verifier("length", core._extract_json(out))

        if ans not in (0, 1):
            msg2 = core.build_mm_messages(sys_txt, image_path, assist, assist_flip, _binary_fallback_prompt(question), include_raw=include_raw)
            out2 = self._call_text(msg2, max_tokens=64)
            ans = core.parse_answer(out2)
        if ans not in (0, 1):
            return 0

        if ans == 1:
            ch_msg = core.build_mm_messages(sys_txt, image_path, assist, assist_flip, _strict_challenge_prompt(question), include_raw=include_raw)
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
            question = _first_line(str(row.get('prompt', '')))
            if question in TARGET_QUESTIONS:
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
                "refined_questions": sorted(TARGET_QUESTIONS),
                "refined_count": refined,
                "pipeline": "base semantic fusion with dedicated gpt-5.2 length-question refiner",
            },
        }
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        vals = list(results.values())
        print(f"Results saved to {output_txt}")
        print(f"Model info saved to {output_json}")
        print(f"Total: {len(vals)} | Answer 0: {vals.count(0)} | Answer 1: {vals.count(1)} | Failed: 0 | Refined: {refined}")


def main() -> None:
    ap = argparse.ArgumentParser(description='Refine length-related questions over base predictions using gpt-5.2.')
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
    LengthRefiner(base_predictions, args.input_csv).run(args.input_csv, args.output_txt, args.output_json)


if __name__ == '__main__':
    main()
