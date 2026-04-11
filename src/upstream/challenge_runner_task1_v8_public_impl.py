#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import io
import os
import sys
from typing import Optional

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TASK1_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
if TASK1_DIR not in sys.path:
    sys.path.insert(0, TASK1_DIR)

from helper_public import run
import challenge_core_v9_public as core
from openai import OpenAI

try:
    from PIL import Image, ImageDraw, ImageEnhance, ImageFont
except Exception:
    Image = None
    ImageDraw = None
    ImageEnhance = None
    ImageFont = None


def _read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""


def _norm_base_url(url: str) -> str:
    u = (url or "").strip().rstrip("/")
    if not u:
        return u
    for suf in ("/chat/completions", "/completions", "/models"):
        if u.endswith(suf):
            u = u[: -len(suf)].rstrip("/")
    if not u.endswith("/v1"):
        u = u + "/v1"
    return u


def _resolve_image_path(image_path: str) -> str:
    p = image_path or ""
    if os.path.isabs(p) and os.path.exists(p):
        return p
    if os.path.exists(p):
        return p
    alt1 = os.path.join(TASK1_DIR, p)
    if os.path.exists(alt1):
        return alt1
    alt2 = os.path.join(os.getcwd(), p)
    if os.path.exists(alt2):
        return alt2
    return p


def _first_line(prompt: str) -> str:
    for ln in str(prompt).replace("\r\n", "\n").split("\n"):
        t = ln.strip()
        if t:
            return t
    return str(prompt).strip()


def _intent(question: str) -> str:
    q = (question or "").lower()
    if "distance" in q and ("a–b" in q or "a-b" in q or "b–c" in q or "b-c" in q):
        return "distance_markers"
    if "vertical columns" in q and "parallel" in q:
        return "parallel_columns"
    if ("red lines" in q and "straight" in q) or ("vertical lines" in q and "straight" in q):
        return "straight_vertical"
    if "diagonal" in q and "aligned" in q:
        return "diagonal_aligned"
    if "straight edges" in q:
        return "straight_edges"
    if "boundary" in q or "border" in q or "adjacent" in q or "adjecent" in q or "regions" in q:
        return "boundary"
    if "color" in q or "same color" in q:
        return "color"
    if "size" in q or "equal in size" in q:
        return "size"
    if "equal length" in q or "same length" in q or "distance" in q:
        return "length"
    if "parallel" in q or "aligned" in q or "straight" in q or "slanted" in q:
        return "straight_generic"
    return "other"


def _intent_to_prim(intent: str) -> str:
    if intent in (
        "parallel_columns",
        "straight_vertical",
        "diagonal_aligned",
        "straight_edges",
        "straight_generic",
    ):
        return "straight"
    if intent == "distance_markers":
        return "length"
    return intent


def _save_png_data_url(im) -> Optional[str]:
    try:
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"
    except Exception:
        return None


def _enhance(im, contrast: float, sharpness: float):
    if ImageEnhance is None:
        return im
    out = ImageEnhance.Contrast(im).enhance(max(1.0, contrast))
    out = ImageEnhance.Sharpness(out).enhance(max(1.0, sharpness))
    return out


def _resize_keep_w(im, target_w: int):
    w, h = im.size
    tw = max(64, int(target_w))
    th = max(64, int(round(h * tw / max(1, w))))
    return im.resize((tw, th), resample=Image.BICUBIC)


def _crop_horizontal_band(im, y_norm: float, half_h_norm: float):
    w, h = im.size
    cy = int(round(h * max(0.0, min(1.0, y_norm))))
    half_h = max(2, int(round(h * max(0.0, half_h_norm))))
    y0 = max(0, cy - half_h)
    y1 = min(h, cy + half_h)
    if y1 <= y0:
        y1 = min(h, y0 + 2)
    return im.crop((0, y0, w, y1))


def _crop_anchor_window(im, x_norm: float, y_norm: float, half_w_norm: float, half_h_norm: float):
    w, h = im.size
    cx = int(round(w * max(0.0, min(1.0, x_norm))))
    cy = int(round(h * max(0.0, min(1.0, y_norm))))
    half_w = max(4, int(round(w * max(0.02, half_w_norm))))
    half_h = max(3, int(round(h * max(0.01, half_h_norm))))
    x0 = max(0, cx - half_w)
    x1 = min(w, cx + half_w)
    y0 = max(0, cy - half_h)
    y1 = min(h, cy + half_h)
    if x1 <= x0:
        x1 = min(w, x0 + 4)
    if y1 <= y0:
        y1 = min(h, y0 + 2)
    return im.crop((x0, y0, x1, y1))


def _label_box(draw, x: int, y: int, text: str, font):
    box_w = 112
    box_h = 18
    draw.rectangle([x, y, x + box_w, y + box_h], fill="white")
    if font is not None:
        draw.text((x + 4, y + 2), text, fill="black", font=font)
    else:
        draw.text((x + 4, y + 2), text, fill="black")


def _build_length_focus_montage(image_path: str) -> Optional[str]:
    if Image is None:
        return None
    try:
        with Image.open(image_path) as im0:
            im0 = _enhance(im0.convert("RGB"), contrast=1.22, sharpness=1.18)
            max_w = max(520, int(core.ASSIST_MAX_W))
            panel_w = max_w // 2
            strip_h = max(150, int(getattr(core, "LENGTH_STRIP_H", 220)))
            end_h = max(84, int(getattr(core, "LENGTH_END_H", 220)))
            top_y = float(getattr(core, "LENGTH_TOP_Y", 0.25))
            bottom_y = float(getattr(core, "LENGTH_BOTTOM_Y", 0.75))
            thin_half = float(getattr(core, "LENGTH_THIN_BAND_HALF_H", 0.012))
            end_half = float(getattr(core, "LENGTH_END_BAND_HALF_H", 0.03))

            full = _resize_keep_w(im0, max_w)
            strip_top = _crop_horizontal_band(im0, top_y, thin_half).resize((max_w, strip_h), resample=Image.BICUBIC)
            strip_bottom = _crop_horizontal_band(im0, bottom_y, thin_half).resize((max_w, strip_h), resample=Image.BICUBIC)

            left_top = _crop_anchor_window(im0, 0.26, top_y, 0.18, end_half).resize((panel_w, end_h), resample=Image.BICUBIC)
            left_bottom = _crop_anchor_window(im0, 0.26, bottom_y, 0.18, end_half).resize((panel_w, end_h), resample=Image.BICUBIC)
            right_top = _crop_anchor_window(im0, 0.74, top_y, 0.18, end_half).resize((max_w - panel_w, end_h), resample=Image.BICUBIC)
            right_bottom = _crop_anchor_window(im0, 0.74, bottom_y, 0.18, end_half).resize((max_w - panel_w, end_h), resample=Image.BICUBIC)

            left_panel = Image.new("RGB", (panel_w, end_h * 2), "white")
            left_panel.paste(left_top, (0, 0))
            left_panel.paste(left_bottom, (0, end_h))
            right_panel = Image.new("RGB", (max_w - panel_w, end_h * 2), "white")
            right_panel.paste(right_top, (0, 0))
            right_panel.paste(right_bottom, (0, end_h))

            pad = 10
            out_h = full.size[1] + pad + strip_top.size[1] + strip_bottom.size[1] + pad + (end_h * 2)
            out = Image.new("RGB", (max_w, out_h), "white")
            y = 0
            out.paste(full, (0, y))
            y += full.size[1] + pad
            out.paste(strip_top, (0, y))
            y += strip_h
            out.paste(strip_bottom, (0, y))
            y += strip_h + pad
            out.paste(left_panel, (0, y))
            out.paste(right_panel, (panel_w, y))

            if ImageDraw is not None:
                draw = ImageDraw.Draw(out)
                font = None
                if ImageFont is not None:
                    try:
                        font = ImageFont.load_default()
                    except Exception:
                        font = None
                _label_box(draw, 4, 4, "FULL VIEW", font)
                _label_box(draw, 4, full.size[1] + pad + 4, "TOP STRIP", font)
                _label_box(draw, 4, full.size[1] + pad + strip_h + 4, "BOTTOM STRIP", font)
                _label_box(draw, 4, full.size[1] + pad + strip_h * 2 + pad + 4, "LEFT END", font)
                _label_box(draw, panel_w + 4, full.size[1] + pad + strip_h * 2 + pad + 4, "RIGHT END", font)
            return _save_png_data_url(out)
    except Exception:
        return None


def _build_color_size_montage(image_path: str) -> Optional[str]:
    if Image is None:
        return None
    try:
        with Image.open(image_path) as im0:
            im0 = _enhance(im0.convert("RGB"), contrast=1.2, sharpness=1.15)
            w, h = im0.size

            max_w = max(480, int(core.ASSIST_MAX_W))
            full = _resize_keep_w(im0, max_w)

            left = im0.crop((0, 0, w // 2, h))
            right = im0.crop((w // 2, 0, w, h))
            col_w = max_w // 2
            col_h = max(220, int(round(col_w * h / max(1, w // 2))))
            left = left.resize((col_w, col_h), resample=Image.BICUBIC)
            right = right.resize((max_w - col_w, col_h), resample=Image.BICUBIC)

            cx0, cx1 = int(w * 0.12), int(w * 0.88)
            c = im0.crop((cx0, 0, cx1, h))
            cw = c.size[0]
            c_left = c.crop((0, 0, cw // 2, h)).resize((col_w, col_h), resample=Image.BICUBIC)
            c_right = c.crop((cw // 2, 0, cw, h)).resize((max_w - col_w, col_h), resample=Image.BICUBIC)

            pad = 8
            out_h = full.size[1] + pad + col_h + pad + col_h
            out = Image.new("RGB", (max_w, out_h), "white")
            y = 0
            out.paste(full, (0, y))
            y += full.size[1] + pad
            out.paste(left, (0, y))
            out.paste(right, (col_w, y))
            y += col_h + pad
            out.paste(c_left, (0, y))
            out.paste(c_right, (col_w, y))
            return _save_png_data_url(out)
    except Exception:
        return None


def _build_vertical_line_montage(image_path: str) -> Optional[str]:
    if Image is None:
        return None
    try:
        with Image.open(image_path) as im0:
            im0 = _enhance(im0.convert("RGB"), contrast=1.22, sharpness=1.2)
            w, h = im0.size
            max_w = max(520, int(core.ASSIST_MAX_W))

            full = _resize_keep_w(im0, max_w)
            rot = im0.transpose(Image.ROTATE_90)
            rot_full = _resize_keep_w(rot, max_w)
            rot_stretch = rot.resize((max_w, max(140, int(rot_full.size[1] * 1.8))), resample=Image.BICUBIC)

            cx0, cx1 = int(w * 0.08), int(w * 0.92)
            center = im0.crop((cx0, 0, cx1, h))
            center = _resize_keep_w(center, max_w)

            pad = 8
            out_h = full.size[1] + pad + center.size[1] + pad + rot_full.size[1] + pad + rot_stretch.size[1]
            out = Image.new("RGB", (max_w, out_h), "white")
            y = 0
            out.paste(full, (0, y))
            y += full.size[1] + pad
            out.paste(center, (0, y))
            y += center.size[1] + pad
            out.paste(rot_full, (0, y))
            y += rot_full.size[1] + pad
            out.paste(rot_stretch, (0, y))
            return _save_png_data_url(out)
    except Exception:
        return None


def _build_diagonal_montage(image_path: str) -> Optional[str]:
    if Image is None:
        return None
    try:
        with Image.open(image_path) as im0:
            im0 = _enhance(im0.convert("RGB"), contrast=1.2, sharpness=1.2)
            w, h = im0.size
            max_w = max(520, int(core.ASSIST_MAX_W))

            full = _resize_keep_w(im0, max_w)
            cx0, cx1 = int(w * 0.12), int(w * 0.88)
            cy0, cy1 = int(h * 0.12), int(h * 0.88)
            center = im0.crop((cx0, cy0, cx1, cy1))
            center = _resize_keep_w(center, max_w)
            rot45 = im0.rotate(45, expand=True, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
            rot45 = _resize_keep_w(rot45, max_w)

            pad = 8
            out_h = full.size[1] + pad + center.size[1] + pad + rot45.size[1]
            out = Image.new("RGB", (max_w, out_h), "white")
            y = 0
            out.paste(full, (0, y))
            y += full.size[1] + pad
            out.paste(center, (0, y))
            y += center.size[1] + pad
            out.paste(rot45, (0, y))
            return _save_png_data_url(out)
    except Exception:
        return None


def _build_square_edge_montage(image_path: str) -> Optional[str]:
    if Image is None:
        return None
    try:
        with Image.open(image_path) as im0:
            im0 = _enhance(im0.convert("RGB"), contrast=1.2, sharpness=1.2)
            w, h = im0.size
            max_w = max(520, int(core.ASSIST_MAX_W))

            full = _resize_keep_w(im0, max_w)
            left = im0.crop((0, 0, w // 2, h))
            right = im0.crop((w // 2, 0, w, h))
            row_h = max(240, int(max_w * h / max(1, w // 2)))
            left = left.resize((max_w // 2, row_h), resample=Image.BICUBIC)
            right = right.resize((max_w - max_w // 2, row_h), resample=Image.BICUBIC)

            pad = 8
            out_h = full.size[1] + pad + row_h
            out = Image.new("RGB", (max_w, out_h), "white")
            out.paste(full, (0, 0))
            y = full.size[1] + pad
            out.paste(left, (0, y))
            out.paste(right, (max_w // 2, y))
            return _save_png_data_url(out)
    except Exception:
        return None


def _build_distance_marker_montage(image_path: str) -> Optional[str]:
    if Image is None:
        return None
    try:
        with Image.open(image_path) as im0:
            im0 = _enhance(im0.convert("RGB"), contrast=1.25, sharpness=1.2)
            w, h = im0.size
            max_w = max(520, int(core.ASSIST_MAX_W))

            full = _resize_keep_w(im0, max_w)
            cx0, cx1 = int(w * 0.16), int(w * 0.84)
            cy0, cy1 = int(h * 0.18), int(h * 0.88)
            center = im0.crop((cx0, cy0, cx1, cy1))
            center = _resize_keep_w(center, max_w)

            lo = center.crop((0, 0, center.size[0] // 2, center.size[1]))
            ro = center.crop((center.size[0] // 2, 0, center.size[0], center.size[1]))
            row_h = max(220, int(round((max_w // 2) * center.size[1] / max(1, center.size[0] // 2))))
            lo = lo.resize((max_w // 2, row_h), resample=Image.BICUBIC)
            ro = ro.resize((max_w - max_w // 2, row_h), resample=Image.BICUBIC)

            pad = 8
            out_h = full.size[1] + pad + center.size[1] + pad + row_h
            out = Image.new("RGB", (max_w, out_h), "white")
            y = 0
            out.paste(full, (0, y))
            y += full.size[1] + pad
            out.paste(center, (0, y))
            y += center.size[1] + pad
            out.paste(lo, (0, y))
            out.paste(ro, (max_w // 2, y))
            return _save_png_data_url(out)
    except Exception:
        return None


def _build_assist(image_path: str, intent: str) -> Optional[str]:
    prim = _intent_to_prim(intent)
    if intent == "length":
        return _build_length_focus_montage(image_path) or core.build_assist_montage(image_path, prim="length")
    if intent in ("straight_vertical", "parallel_columns"):
        return _build_vertical_line_montage(image_path)
    if intent == "diagonal_aligned":
        return _build_diagonal_montage(image_path)
    if intent == "straight_edges":
        return _build_square_edge_montage(image_path)
    if intent == "distance_markers":
        return _build_distance_marker_montage(image_path)
    if prim in ("length", "straight", "boundary"):
        return core.build_assist_montage(image_path, prim=prim)
    if prim in ("color", "size"):
        return _build_color_size_montage(image_path)
    return core.build_assist_montage(image_path, prim="other")


def _bool_env(name: str, default: bool) -> bool:
    v = os.getenv(name, "1" if default else "0").strip().lower()
    return v not in ("0", "false", "no", "off")


STRICT_CHALLENGE = _bool_env("VQA_V8_STRICT_CHALLENGE", False)
INCLUDE_RAW_ALWAYS = _bool_env("VQA_V8_INCLUDE_RAW_ALWAYS", False)
CHALLENGE_ENABLE = _bool_env("VQA_V8_CHALLENGE_ENABLE", True)


def _challenge_uncertain_rule() -> str:
    return "answer 0" if STRICT_CHALLENGE else "answer 1"


def _verifier_prompt(question: str, intent: str) -> str:
    prim = _intent_to_prim(intent)
    if intent == "distance_markers":
        return f"""
You are verifying objective spacing in the drawing.
Compare the horizontal gap A–B and B–C using the vertical marker lines as anchors.
Ignore text labels and any global illusion prior.
If the two gaps are clearly different, answer NO; otherwise YES.
Return ONLY: <answer>1</answer> or <answer>0</answer>.

Question:
{question}
""".strip()

    if intent in ("straight_vertical", "parallel_columns"):
        return f"""
You are verifying objective vertical-line geometry.
Use both original and rotated views (in rotated view, original vertical lines become horizontal).
- For "straight": if any target line has visible bend/kink, answer NO.
- For "parallel": if target columns do not keep the same direction (converge/diverge), answer NO.
- If no such violation is visible, answer YES.
Return ONLY: <answer>1</answer> or <answer>0</answer>.

Question:
{question}
""".strip()

    if intent == "diagonal_aligned":
        return f"""
You are verifying objective alignment of the red and black solid diagonal lines.
Judge whether the referenced solid lines are actually aligned/collinear as drawn.
If there is clear offset/misalignment, answer NO; otherwise YES.
Return ONLY: <answer>1</answer> or <answer>0</answer>.

Question:
{question}
""".strip()

    if intent == "straight_edges":
        return f"""
You are verifying objective edge straightness of the left and right squares.
Inspect each edge directly; if any referenced edge appears bent/curved, answer NO.
If edges are straight as drawn, answer YES.
Return ONLY: <answer>1</answer> or <answer>0</answer>.

Question:
{question}
""".strip()

    if prim == "length":
        return f"""
You are verifying objective geometry in an illusion-like image.
Use the strip and endpoint panels to compare ONLY the referenced segments.
- Check LEFT endpoints and RIGHT endpoints separately.
- In the endpoint panels, the top and bottom shafts should begin and end at matching positions for YES.
- If either end appears mismatched, answer NO.
Return ONLY: <answer>1</answer> or <answer>0</answer>.

Question:
{question}
""".strip()

    if prim == "straight":
        return f"""
You are verifying objective line geometry (straight/parallel/aligned).
Inspect target lines only; ignore distracting background textures.
If any target line is visibly bent or misaligned for the asked relation, answer NO.
Otherwise answer YES.
Return ONLY: <answer>1</answer> or <answer>0</answer>.

Question:
{question}
""".strip()

    if prim == "boundary":
        return f"""
You are verifying explicit boundaries.
Interpret boundary as an explicit separating border/edge between adjacent regions (not just color change).
If the statement says EVERY adjacent pair and you find one counterexample, answer NO.
If no such counterexample is visible, answer YES.
Return ONLY: <answer>1</answer> or <answer>0</answer>.

Question:
{question}
""".strip()

    if prim == "color":
        return f"""
You are verifying objective color sameness.
Judge actual drawn fill color of the referenced targets, not contextual appearance illusions.
If there is a clear color mismatch, answer NO; otherwise YES.
Return ONLY: <answer>1</answer> or <answer>0</answer>.

Question:
{question}
""".strip()

    if prim == "size":
        return f"""
You are verifying objective size equality.
Judge actual geometric size of referenced targets as drawn, not perceived size illusion.
If there is a clear size mismatch, answer NO; otherwise YES.
Return ONLY: <answer>1</answer> or <answer>0</answer>.

Question:
{question}
""".strip()

    return f"""
Answer the binary visual question using objective properties shown in the image.
Return ONLY: <answer>1</answer> or <answer>0</answer>.

Question:
{question}
""".strip()


def _challenge_prompt(question: str, intent: str) -> str:
    prim = _intent_to_prim(intent)
    uncertain_rule = _challenge_uncertain_rule()
    if intent == "length":
        return f"""
Act as a strict challenger and try to REFUTE answer=1.
Check endpoint alignment of the two compared segments.
If either endpoint pair appears clearly mismatched, answer 0.
If uncertain, {uncertain_rule}.
Otherwise answer 1.
Return ONLY: <answer>1</answer> or <answer>0</answer>.

Question:
{question}
""".strip()

    if intent == "distance_markers":
        return f"""
Act as a strict challenger and try to REFUTE answer=1.
Try to find concrete evidence that A–B and B–C are not equal in spacing.
If you find a clear mismatch, answer 0.
If uncertain, {uncertain_rule}.
Otherwise answer 1.
Return ONLY: <answer>1</answer> or <answer>0</answer>.

Question:
{question}
""".strip()

    if intent in ("straight_vertical", "parallel_columns", "diagonal_aligned", "straight_edges"):
        return f"""
Act as a strict challenger and try to REFUTE answer=1.
Find a concrete visible counterexample to the claim (bend, misalignment, non-parallel relation, or non-straight edge).
If you find one clear counterexample, answer 0.
If uncertain, {uncertain_rule}.
Otherwise answer 1.
Return ONLY: <answer>1</answer> or <answer>0</answer>.

Question:
{question}
""".strip()

    if prim == "boundary":
        return f"""
Act as a strict challenger and try to REFUTE answer=1.
Check whether the statement "boundary in between every adjacent regions" truly holds.
If you find one violating adjacent pair (no explicit boundary), answer 0.
If you are uncertain, {uncertain_rule}.
Otherwise answer 1.
Return ONLY: <answer>1</answer> or <answer>0</answer>.

Question:
{question}
""".strip()

    return f"""
Act as a strict challenger and try to REFUTE answer=1.
Find a concrete visible counterexample to the claim in the question.
- If a clear counterexample exists, answer 0.
- If no clear counterexample is visible, answer 1.
Return ONLY: <answer>1</answer> or <answer>0</answer>.

Question:
{question}
""".strip()


def _need_challenge(intent: str) -> bool:
    if not CHALLENGE_ENABLE:
        return False
    default_intents = {
        "length",
        "distance_markers",
        "straight_vertical",
        "parallel_columns",
        "diagonal_aligned",
        "straight_generic",
        "boundary",
    }
    custom = os.getenv("VQA_V8_CHALLENGE_INTENTS", "").strip()
    if not custom:
        return intent in default_intents
    intents = {x.strip() for x in custom.split(",") if x.strip()}
    return intent in intents


# ---- safer default runtime config for this wrapper ----
_default_key = (
    os.getenv("VQA_API_KEY", "").strip()
    or os.getenv("OPENAI_API_KEY", "").strip()
    or _read_text(os.path.join(REPO_ROOT, ".openai_api_key"))
)
if _default_key:
    core.API_KEY = _default_key
core.BASE_URL = _norm_base_url(os.getenv("VQA_BASE_URL", "https://api.xbai.top/v1"))
core.JUDGE_MODEL = (os.getenv("VQA_JUDGE_MODEL", "gpt-5.1") or "gpt-5.1").strip()
core.FALLBACK_MODEL = (os.getenv("VQA_FALLBACK_MODEL", "gpt-5") or "gpt-5").strip()
core.LENGTH_USE_ICL = False
REQUEST_TIMEOUT = float(os.getenv("VQA_REQUEST_TIMEOUT", "45"))


class MySolver(core.MySolver):
    def __init__(self):
        super().__init__()
        try:
            self.client = OpenAI(
                api_key=(core.API_KEY or None),
                base_url=(core.BASE_URL or None),
                timeout=REQUEST_TIMEOUT,
                max_retries=0,
            )
            print(f"REQUEST_TIMEOUT={REQUEST_TIMEOUT}")
        except Exception:
            pass

    def solve(self, image_path: str, prompt: str) -> int:
        image_path = _resolve_image_path(image_path)
        q = _first_line(prompt)
        intent = _intent(q)
        prim = _intent_to_prim(intent)
        sys_txt = core.system_text()

        assist = _build_assist(image_path, intent)
        assist_flip = core.maybe_flip_data_url(assist) if (assist and prim in ("length", "straight")) else None
        include_raw = INCLUDE_RAW_ALWAYS or (prim == "boundary") or (assist is None)

        user_txt = _verifier_prompt(q, intent)
        msg = core.build_mm_messages(
            sys_txt,
            image_path,
            assist,
            assist_flip,
            user_txt,
            include_raw=include_raw,
        )
        out = self._call(core.JUDGE_MODEL, msg, max_tokens=max(core.JUDGE_MAX_TOKENS, 180)) or ""
        ans = core.parse_answer(out)
        if ans in (0, 1):
            if ans == 1 and _need_challenge(intent):
                ch_msg = core.build_mm_messages(
                    sys_txt,
                    image_path,
                    assist,
                    assist_flip,
                    _challenge_prompt(q, intent),
                    include_raw=include_raw,
                )
                ch_out = self._call(core.JUDGE_MODEL, ch_msg, max_tokens=96) or ""
                ch = core.parse_answer(ch_out)
                if ch == 0:
                    return 0
            return ans

        retry = (
            "Return ONLY one line exactly: <answer>1</answer> or <answer>0</answer>.\n\n"
            f"Question:\n{q}"
        )
        msg2 = core.build_mm_messages(
            sys_txt,
            image_path,
            assist,
            assist_flip,
            retry,
            include_raw=include_raw,
        )
        out2 = self._call(core.JUDGE_MODEL, msg2, max_tokens=64) or ""
        ans2 = core.parse_answer(out2)
        if ans2 in (0, 1):
            if ans2 == 1 and _need_challenge(intent):
                ch_msg2 = core.build_mm_messages(
                    sys_txt,
                    image_path,
                    assist,
                    assist_flip,
                    _challenge_prompt(q, intent),
                    include_raw=include_raw,
                )
                ch_out2 = self._call(core.JUDGE_MODEL, ch_msg2, max_tokens=96) or ""
                ch2 = core.parse_answer(ch_out2)
                if ch2 == 0:
                    return 0
            return ans2

        return 0

    def model_info(self) -> dict:
        return {
            "model": core.JUDGE_MODEL,
            "parameters": {
                "base_url": core.BASE_URL,
                "judge_model": core.JUDGE_MODEL,
                "fallback_model": core.FALLBACK_MODEL,
                "temperature": core.TEMPERATURE,
                "top_p": core.TOP_P,
                "seed": core.SEED,
                "pipeline": "v8 intent-specific assist + fixed qualitative length montage + selective challenger on high-risk intents",
                "no_minus_one": True,
            },
        }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Task1 public runner v8 (fixed qualitative length montage + selective challenger, no -1)."
    )
    ap.add_argument("--input-csv", default=os.path.join(TASK1_DIR, "test.csv"))
    ap.add_argument("--output-txt", default=os.path.join(THIS_DIR, "predictions_public_v8.txt"))
    ap.add_argument("--output-json", default=os.path.join(THIS_DIR, "model_public_v8.json"))
    args = ap.parse_args()
    run(MySolver(), args.input_csv, args.output_txt, args.output_json)
