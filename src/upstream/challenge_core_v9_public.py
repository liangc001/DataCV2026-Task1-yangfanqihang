#!/usr/bin/env python3
"""
Task I (Classic Illusion Understanding) — Verifier v9: length ICL (synthetic calibration) + strict verifier (rules-compliant)

Why this runner exists
----------------------
On classic illusion questions, many VLMs collapse to an "illusion prior":
they recognize the pattern and answer the canonical fact (often "YES"),
which fails badly on *perturbed* images where the canonical fact is broken.

This runner tries to break that failure mode by:
1) Presentation-only "montage" views (length: blend + end-zooms; straight: multi-candidate bands; boundary: raw-focused).
2) A strict VERIFIER checklist that must be answered based on the montage (no measurement).

Compared to v5:
- Keeps boundary=RAW-only and straight=multi-candidate bands (same as v5 baseline).
- Length: adds a tiny in-context learning (ICL) calibration using *synthetic* Muller-Lyer-like examples
  (both equal and not-equal) to counter the model's strong illusion prior / bias.
- Still single-call per sample (no separate advocate/judge roles).

Rules compliance (Task I)
-------------------------
- No training / fine-tuning.
- No deterministic measurement or computation (no pixel statistics, no ruler/grid quantification,
  no explicit length/angle/distance estimation used as decision rules).
- Only inference-time prompting + basic presentation aids (crop/zoom/resize/contrast/sharpness/flip).

This file is an alternative runner; it does NOT replace other runners.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from helper_public import Solver, run

try:
    from PIL import Image, ImageDraw, ImageEnhance, ImageFont
except Exception:  # pragma: no cover
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]
    ImageEnhance = None  # type: ignore[assignment]
    ImageFont = None  # type: ignore[assignment]


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------- API config ----------------
def _normalize_base_url(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return u
    u = u.rstrip("/")
    for suf in ("/chat/completions", "/completions", "/models"):
        if u.endswith(suf):
            u = u[: -len(suf)].rstrip("/")
    if not u.endswith("/v1"):
        u = u + "/v1"
    return u


def _normalize_model_name(name: str) -> str:
    n = (name or "").strip()
    return n[:-3] if n.endswith("/v1") else n


BASE_URL = _normalize_base_url(os.getenv("VQA_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
# Never rely on a hardcoded real key in code; set via env var.
API_KEY = os.getenv("VQA_API_KEY", "")

# Models confirmed available in this environment (per user):
#   qwen3-vl-plus
#   qwen3-vl-plus-2025-12-19
#   qwen3-vl-plus-2025-09-23
# v9 is a single-call verifier pipeline; keep YES/NO envs for compatibility,
# but only JUDGE_MODEL is used unless you edit the code.
YES_MODEL = _normalize_model_name(os.getenv("VQA_YES_MODEL", "qwen3-vl-plus"))
NO_MODEL = _normalize_model_name(os.getenv("VQA_NO_MODEL", "qwen3-vl-plus"))
JUDGE_MODEL = _normalize_model_name(os.getenv("VQA_JUDGE_MODEL", "qwen3-vl-plus"))
FALLBACK_MODEL = _normalize_model_name(os.getenv("VQA_FALLBACK_MODEL", "qwen3-vl-plus-2025-09-23"))

TEMPERATURE = float(os.getenv("VQA_TEMPERATURE", "0.0"))
TOP_P = float(os.getenv("VQA_TOP_P", "1.0"))
SEED = int(os.getenv("VQA_SEED", "42"))

ADVOCATE_MAX_TOKENS = int(os.getenv("VQA_ADVOCATE_MAX_TOKENS", "260"))
JUDGE_MAX_TOKENS = int(os.getenv("VQA_JUDGE_MAX_TOKENS", "120"))

# Optional debug dump directory: saves assist montage + advocate/judge texts for inspection.
DEBUG_DIR = os.getenv("VQA_DEBUG_DIR", "").strip()


# ---------------- Presentation-only assist knobs ----------------
USE_ASSIST_VIEW = os.getenv("VQA_USE_ASSIST_VIEW", "1") != "0"
USE_FLIP_VIEW = os.getenv("VQA_USE_FLIP_VIEW", "1") != "0"

ASSIST_MAX_W = int(os.getenv("VQA_ASSIST_MAX_W", "768"))
ASSIST_CONTRAST = float(os.getenv("VQA_ASSIST_CONTRAST", "1.35"))
ASSIST_SHARPNESS = float(os.getenv("VQA_ASSIST_SHARPNESS", "1.25"))

BOUNDARY_CONTRAST = float(os.getenv("VQA_BOUNDARY_CONTRAST", "7.5"))
BOUNDARY_SHARPNESS = float(os.getenv("VQA_BOUNDARY_SHARPNESS", "1.9"))

# Fixed-layout band positions (still presentation-only; no computation).
LENGTH_TOP_Y = float(os.getenv("VQA_LENGTH_TOP_Y", "0.25"))
LENGTH_BOTTOM_Y = float(os.getenv("VQA_LENGTH_BOTTOM_Y", "0.75"))
LENGTH_BAND_HALF_H = float(os.getenv("VQA_LENGTH_BAND_HALF_H", "0.055"))
# A thinner vertical crop suppresses slanted fins/arrow tips, reducing illusion priors.
LENGTH_THIN_BAND_HALF_H = float(os.getenv("VQA_LENGTH_THIN_BAND_HALF_H", "0.012"))
# A slightly thicker crop for endpoint zooms keeps the fin/shaft junction corner visible.
LENGTH_END_BAND_HALF_H = float(os.getenv("VQA_LENGTH_END_BAND_HALF_H", "0.03"))
# Horizontal zoom window for length (reduces white margins so endpoints are larger).
LENGTH_X0 = float(os.getenv("VQA_LENGTH_X0", "0.2"))
LENGTH_X1 = float(os.getenv("VQA_LENGTH_X1", "0.8"))
LENGTH_STRIP_H = int(os.getenv("VQA_LENGTH_STRIP_H", "220"))
LENGTH_END_H = int(os.getenv("VQA_LENGTH_END_H", "220"))
# Length ICL (synthetic calibration) toggle.
LENGTH_USE_ICL = os.getenv("VQA_LENGTH_USE_ICL", "1") != "0"
LENGTH_ICL_PRESET = os.getenv("VQA_LENGTH_ICL_PRESET", "4").strip()

STRAIGHT_TOP_Y = float(os.getenv("VQA_STRAIGHT_TOP_Y", "0.40"))
STRAIGHT_BOTTOM_Y = float(os.getenv("VQA_STRAIGHT_BOTTOM_Y", "0.70"))
STRAIGHT_BAND_HALF_H = float(os.getenv("VQA_STRAIGHT_BAND_HALF_H", "0.075"))
STRAIGHT_STRIP_H = int(os.getenv("VQA_STRAIGHT_STRIP_H", "170"))
STRAIGHT_STRETCH = float(os.getenv("VQA_STRAIGHT_STRETCH", "3.0"))

BOUNDARY_PATCH_H = int(os.getenv("VQA_BOUNDARY_PATCH_H", "240"))
BOUNDARY_STRIP_HALF_W = float(os.getenv("VQA_BOUNDARY_STRIP_HALF_W", "0.06"))

# Straight: use multiple candidate band positions to avoid brittle fixed crops.
STRAIGHT_CAND_TOP = os.getenv("VQA_STRAIGHT_CAND_TOP", "0.32,0.36,0.40,0.44")
STRAIGHT_CAND_BOTTOM = os.getenv("VQA_STRAIGHT_CAND_BOTTOM", "0.56,0.60,0.64,0.68")
STRAIGHT_CAND_HALF_H = float(os.getenv("VQA_STRAIGHT_CAND_HALF_H", "0.022"))
STRAIGHT_CAND_PANEL_H = int(os.getenv("VQA_STRAIGHT_CAND_PANEL_H", "150"))
GUIDE_LINE_COLOR = tuple(int(x) for x in os.getenv("VQA_GUIDE_LINE_RGB", "220,40,40").split(",")[:3])


# ---------------- Parsing ----------------
ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)


def parse_answer(text: Optional[str]) -> int:
    if not text:
        return -1
    s = text.strip()
    m = ANSWER_TAG_RE.search(s)
    if m:
        a = m.group(1).strip().lower()
        if a in ("1", "yes", "y", "true"):
            return 1
        if a in ("0", "no", "n", "false"):
            return 0
    last = s.splitlines()[-1].strip().lower()
    if last in ("<answer>1</answer>", "1", "yes"):
        return 1
    if last in ("<answer>0</answer>", "0", "no"):
        return 0
    return -1


def _extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    t = text.strip()
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _norm_text(v: Any) -> str:
    try:
        s = str(v).strip().lower()
        # Make common variants match our expected tokens.
        s = s.replace("-", "_").replace(" ", "_")
        while "__" in s:
            s = s.replace("__", "_")
        return s
    except Exception:
        return ""


def _as_int01(v: Any) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, bool):
        return 1 if v else 0
    s = _norm_text(v)
    if s in ("1", "yes", "y", "true"):
        return 1
    if s in ("0", "no", "n", "false"):
        return 0
    try:
        f = float(s)
        if f.is_integer() and int(f) in (0, 1):
            return int(f)
    except Exception:
        pass
    return None


def derive_answer_from_verifier(prim: str, obj: Optional[dict]) -> int:
    """
    Convert verifier JSON into the final 0/1 answer using qualitative logic only.
    """
    if not isinstance(obj, dict):
        return -1

    prim = (prim or "").strip().lower()

    if prim == "length":
        # Accept both v4 keys ("*_junction") and v5 keys ("*_end").
        left = _norm_text(obj.get("left_end") if obj.get("left_end") is not None else obj.get("left_junction"))
        right = _norm_text(obj.get("right_end") if obj.get("right_end") is not None else obj.get("right_junction"))

        def _is_not_aligned(tok: str) -> bool:
            # Common variants: not_aligned, misaligned, not_align, does_not_align, etc.
            return ("misalign" in tok) or (("not" in tok or tok == "no") and "align" in tok) or tok in ("not_align",)

        def _is_aligned(tok: str) -> bool:
            if tok in ("align", "aligned", "yes", "true"):
                return True
            return ("align" in tok) and (not _is_not_aligned(tok))

        if _is_not_aligned(left) or _is_not_aligned(right):
            return 0
        if _is_aligned(left) and _is_aligned(right):
            return 1
        bg = _as_int01(obj.get("best_guess"))
        return bg if bg in (0, 1) else -1

    if prim == "straight":
        top = _norm_text(obj.get("top_line"))
        bottom = _norm_text(obj.get("bottom_line"))
        if top in ("straight_line", "straightness_ok", "straighten"):
            top = "straight"
        if bottom in ("straight_line", "straightness_ok", "straighten"):
            bottom = "straight"
        if top == "not_straight" or bottom == "not_straight":
            return 0
        if top == "straight" and bottom == "straight":
            return 1
        bg = _as_int01(obj.get("best_guess"))
        return bg if bg in (0, 1) else -1

    if prim == "boundary":
        region = _norm_text(obj.get("region_type"))
        if region == "single" or region.startswith("single_"):
            return 1
        if region == "multiple" or region.startswith("multiple_"):
            borders = _norm_text(obj.get("explicit_borders_everywhere"))
            if borders == "yes":
                return 1
            if borders == "no":
                return 0
        bg = _as_int01(obj.get("best_guess"))
        return bg if bg in (0, 1) else -1

    bg = _as_int01(obj.get("best_guess"))
    return bg if bg in (0, 1) else -1


# ---------------- Image helpers ----------------
def encode_image_to_data_url(image_path: str) -> str:
    ext = os.path.splitext(image_path)[1].lower()
    media = "image/png" if ext == ".png" else "image/jpeg"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{media};base64,{b64}"


def encode_bytes_to_data_url(img_bytes: bytes, media: str) -> str:
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:{media};base64,{b64}"


def _save_png_to_data_url(im) -> str:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return encode_bytes_to_data_url(buf.getvalue(), "image/png")


def _enhance(im, contrast: float, sharpness: float):
    if ImageEnhance is None:
        return im
    out = im
    out = ImageEnhance.Contrast(out).enhance(max(1.0, float(contrast)))
    out = ImageEnhance.Sharpness(out).enhance(max(1.0, float(sharpness)))
    return out


def _draw_muller_lyer(
    shaft_top: int,
    shaft_bottom: int,
    *,
    size: Tuple[int, int] = (512, 256),
    stroke: int = 2,
    fin: int = 18,
) -> Optional[Any]:
    """
    Create a simple Muller-Lyer-like synthetic image for ICL calibration.
    Purely presentation: no measurement, no dataset leakage.
    """
    if Image is None or ImageDraw is None:
        return None
    w, h = size
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)

    cx = w // 2
    y_top = int(round(h * 0.23))
    y_bot = int(round(h * 0.70))

    def draw_one(y: int, shaft_len: int, fins_outward: bool) -> None:
        x0 = cx - int(round(shaft_len / 2.0))
        x1 = cx + int(round(shaft_len / 2.0))
        d.line([(x0, y), (x1, y)], fill="black", width=stroke)

        def fins_at(x: int, direction: int) -> None:
            # direction: -1 for left end, +1 for right end
            tip = x + (direction * fin if fins_outward else -direction * fin)
            d.line([(x, y), (tip, y - fin)], fill="black", width=stroke)
            d.line([(x, y), (tip, y + fin)], fill="black", width=stroke)

        fins_at(x0, -1)
        fins_at(x1, +1)

    # Top: outward fins; Bottom: inward fins (matches common benchmark layout).
    draw_one(y_top, shaft_top, fins_outward=True)
    draw_one(y_bot, shaft_bottom, fins_outward=False)
    return im


def build_length_icl_examples() -> List[Tuple[str, int, str]]:
    """
    Return a small set of (name, answer, data_url) synthetic examples for length ICL.
    The preset can be controlled via env VQA_LENGTH_ICL_PRESET:
      - \"2\": equal+notequal (short)
      - \"4\": equal+notequal (short) + equal+notequal (long)
    """
    preset = (LENGTH_ICL_PRESET or "4").strip()
    # (name, answer, top_len, bottom_len)
    if preset == "2":
        specs = [
            ("EQUAL (short)", 1, 150, 150),
            ("NOT EQUAL (short)", 0, 190, 120),
        ]
    else:
        specs = [
            ("EQUAL (short)", 1, 150, 150),
            ("NOT EQUAL (short)", 0, 190, 120),
            ("EQUAL (long)", 1, 320, 320),
            ("NOT EQUAL (long)", 0, 380, 260),
        ]

    out: List[Tuple[str, int, str]] = []
    for name, ans, top_len, bot_len in specs:
        im = _draw_muller_lyer(top_len, bot_len)
        if im is None:
            continue
        out.append((name, ans, _save_png_to_data_url(im)))
    return out


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _crop_band_full_width(im, y_norm: float, half_h_norm: float):
    w, h = im.size
    y = int(round(_clamp01(y_norm) * h))
    hh = int(round(max(2.0, _clamp01(half_h_norm) * h)))
    y0 = max(0, y - hh)
    y1 = min(h, y + hh)
    if y1 <= y0 + 2:
        y0 = max(0, y - 6)
        y1 = min(h, y + 6)
    return im.crop((0, y0, w, y1))


def _resize_to_width(im, target_w: int):
    w, h = im.size
    tw = int(max(32, target_w))
    if w == tw:
        return im
    th = int(round(h * (tw / float(w))))
    return im.resize((tw, max(32, th)), resample=Image.BICUBIC)


def _pad_to_size(im, w: int, h: int, bg: str = "white"):
    out = Image.new("RGB", (w, h), bg)
    x = max(0, (w - im.size[0]) // 2)
    y = max(0, (h - im.size[1]) // 2)
    out.paste(im, (x, y))
    return out


def _ensure_debug_dir():
    if not DEBUG_DIR:
        return None
    try:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        return DEBUG_DIR
    except Exception:
        return None


def _debug_write(path: str, data: bytes, mode: str = "wb") -> None:
    try:
        with open(path, mode) as f:
            f.write(data)  # type: ignore[arg-type]
    except Exception:
        return


def _debug_write_text(path: str, text: str) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception:
        return


def build_assist_montage(image_path: str, prim: str) -> Optional[str]:
    """
    Build a single composite assist image as a PNG data URL.
    Presentation-only: fixed crops + resize + contrast/sharpness + optional stretching.
    """
    if (not USE_ASSIST_VIEW) or (Image is None):
        return None
    try:
        with Image.open(image_path) as im0:
            im0 = im0.convert("RGB")

            # Boundary: RAW-only (no enhanced montage). Heavy contrast can introduce banding artifacts
            # on near-uniform images and trick the verifier into hallucinating multiple regions.
            if prim == "boundary":
                return None

            # Mild enhancement for geometry visibility (length/straight).
            base = _enhance(im0, ASSIST_CONTRAST, ASSIST_SHARPNESS)

            panel_w = max(192, ASSIST_MAX_W // 2)

            if prim == "length":
                # Use a thinner vertical crop for the main strips to suppress the arrowhead tips/slanted fins.
                band_top_strip = _crop_band_full_width(base, LENGTH_TOP_Y, LENGTH_THIN_BAND_HALF_H)
                band_bot_strip = _crop_band_full_width(base, LENGTH_BOTTOM_Y, LENGTH_THIN_BAND_HALF_H)
                strip_top = band_top_strip.resize((ASSIST_MAX_W, LENGTH_STRIP_H), resample=Image.BICUBIC)
                strip_bot = band_bot_strip.resize((ASSIST_MAX_W, LENGTH_STRIP_H), resample=Image.BICUBIC)

                # Use a slightly thicker crop for endpoint zooms so the fin/shaft junction corner remains visible.
                band_top_end = _crop_band_full_width(base, LENGTH_TOP_Y, LENGTH_END_BAND_HALF_H)
                band_bot_end = _crop_band_full_width(base, LENGTH_BOTTOM_Y, LENGTH_END_BAND_HALF_H)
                end_top = band_top_end.resize((ASSIST_MAX_W, LENGTH_STRIP_H), resample=Image.BICUBIC)
                end_bot = band_bot_end.resize((ASSIST_MAX_W, LENGTH_STRIP_H), resample=Image.BICUBIC)

                # Endpoint-focused crops: left half and right half.
                def split_lr(strip):
                    left = strip.crop((0, 0, panel_w, strip.size[1]))
                    right = strip.crop((ASSIST_MAX_W - panel_w, 0, ASSIST_MAX_W, strip.size[1]))
                    return left, right

                top_l, top_r = split_lr(end_top)
                bot_l, bot_r = split_lr(end_bot)

                end_h = max(64, int(LENGTH_END_H))
                top_l = top_l.resize((panel_w, end_h), resample=Image.BICUBIC)
                top_r = top_r.resize((panel_w, end_h), resample=Image.BICUBIC)
                bot_l = bot_l.resize((panel_w, end_h), resample=Image.BICUBIC)
                bot_r = bot_r.resize((panel_w, end_h), resample=Image.BICUBIC)

                left_panel = Image.new("RGB", (panel_w, end_h * 2), "white")
                left_panel.paste(top_l, (0, 0))
                left_panel.paste(bot_l, (0, end_h))
                right_panel = Image.new("RGB", (panel_w, end_h * 2), "white")
                right_panel.paste(top_r, (0, 0))
                right_panel.paste(bot_r, (0, end_h))

                # Compose: 2 strips + endpoint zooms. (Avoid blend-row confusion.)
                gap = 10
                comp_h = (LENGTH_STRIP_H * 2) + gap + (end_h * 2)
                comp = Image.new("RGB", (ASSIST_MAX_W, comp_h), "white")
                y = 0
                comp.paste(strip_top, (0, y))
                y += LENGTH_STRIP_H
                comp.paste(strip_bot, (0, y))
                y += LENGTH_STRIP_H + gap
                comp.paste(left_panel, (0, y))
                comp.paste(right_panel, (panel_w, y))

                # Small labels to reduce panel confusion.
                if ImageDraw is not None:
                    draw = ImageDraw.Draw(comp)
                    font = None
                    if ImageFont is not None:
                        try:
                            font = ImageFont.load_default()
                        except Exception:
                            font = None

                    def label(x: int, y: int, text: str):
                        draw.rectangle([x, y, x + 92, y + 18], fill="white")
                        if font is not None:
                            draw.text((x + 4, y + 2), text, fill="black", font=font)
                        else:
                            draw.text((x + 4, y + 2), text, fill="black")

                    label(4, 4, "TOP")
                    label(4, LENGTH_STRIP_H + 4, "BOTTOM")
                    label(4, (LENGTH_STRIP_H * 2) + gap + 4, "LEFT END")
                    label(panel_w + 4, (LENGTH_STRIP_H * 2) + gap + 4, "RIGHT END")

                return _save_png_to_data_url(comp)

            if prim == "straight":
                # Multi-candidate bands reduce brittleness of fixed y-crops.
                def parse_list(s: str) -> List[float]:
                    out: List[float] = []
                    for part in (s or "").split(","):
                        part = part.strip()
                        if not part:
                            continue
                        try:
                            out.append(float(part))
                        except Exception:
                            continue
                    return out

                top_ys = parse_list(STRAIGHT_CAND_TOP) or [0.32, 0.36, 0.40, 0.44]
                bot_ys = parse_list(STRAIGHT_CAND_BOTTOM) or [0.56, 0.60, 0.64, 0.68]

                panels: List[Tuple[str, Any]] = []
                for i, y_norm in enumerate(top_ys, start=1):
                    band = _crop_band_full_width(base, y_norm, STRAIGHT_CAND_HALF_H)
                    panel = band.resize((ASSIST_MAX_W, STRAIGHT_CAND_PANEL_H), resample=Image.BICUBIC)
                    panels.append((f"T{i}", panel))
                for i, y_norm in enumerate(bot_ys, start=1):
                    band = _crop_band_full_width(base, y_norm, STRAIGHT_CAND_HALF_H)
                    panel = band.resize((ASSIST_MAX_W, STRAIGHT_CAND_PANEL_H), resample=Image.BICUBIC)
                    panels.append((f"B{i}", panel))

                gap = 8
                total_h = len(panels) * STRAIGHT_CAND_PANEL_H + (len(panels) - 1) * gap
                comp = Image.new("RGB", (ASSIST_MAX_W, total_h), "white")
                draw = ImageDraw.Draw(comp) if ImageDraw is not None else None
                font = None
                if ImageFont is not None:
                    try:
                        font = ImageFont.load_default()
                    except Exception:
                        font = None

                y = 0
                for tag, panel in panels:
                    comp.paste(panel, (0, y))
                    # Reference guideline at the vertical center of the panel (qualitative aid).
                    if draw is not None:
                        midy = y + STRAIGHT_CAND_PANEL_H // 2
                        try:
                            r, g, b = GUIDE_LINE_COLOR
                        except Exception:
                            r, g, b = (220, 40, 40)
                        draw.line([(0, midy), (ASSIST_MAX_W, midy)], fill=(r, g, b), width=2)
                        # Tag label (T1/T2/.../B1/...) for the verifier to reference.
                        if font is not None:
                            draw.rectangle([0, y, 42, y + 18], fill="white")
                            draw.text((4, y + 2), tag, fill="black", font=font)
                    y += STRAIGHT_CAND_PANEL_H + gap

                return _save_png_to_data_url(comp)

            if prim == "boundary":
                w, h = base.size
                # Row 1: three vertical-strip zooms to reveal whether the image has discrete regions (e.g., stripes)
                # and whether there is an *explicit* border line between them.
                half_w = max(1, int(round(w * _clamp01(BOUNDARY_STRIP_HALF_W))))
                centers = [int(round(w * 0.25)), int(round(w * 0.50)), int(round(w * 0.75))]
                strips = []
                for cx in centers:
                    x0 = max(0, cx - half_w)
                    x1 = min(w, cx + half_w)
                    if x1 <= x0 + 1:
                        x0 = max(0, cx - 2)
                        x1 = min(w, cx + 2)
                    strips.append(base.crop((x0, 0, x1, h)))

                col_w = max(96, ASSIST_MAX_W // 3)
                strip_h = int(max(96, BOUNDARY_PATCH_H))
                strip_row = Image.new("RGB", (ASSIST_MAX_W, strip_h), "white")
                xs = [0, col_w, col_w * 2]
                for i, s in enumerate(strips[:3]):
                    ww = col_w if i < 2 else (ASSIST_MAX_W - col_w * 2)
                    strip_row.paste(s.resize((ww, strip_h), resample=Image.BICUBIC), (xs[i], 0))

                # Rows 2-3: 2x2 quadrant zooms for systematic scanning.
                quads = [
                    base.crop((0, 0, w // 2, h // 2)),
                    base.crop((w // 2, 0, w, h // 2)),
                    base.crop((0, h // 2, w // 2, h)),
                    base.crop((w // 2, h // 2, w, h)),
                ]
                quads_r = [q.resize((panel_w, strip_h), resample=Image.BICUBIC) for q in quads]
                quad_canvas = Image.new("RGB", (ASSIST_MAX_W, strip_h * 2), "white")
                quad_canvas.paste(quads_r[0], (0, 0))
                quad_canvas.paste(quads_r[1], (panel_w, 0))
                quad_canvas.paste(quads_r[2], (0, strip_h))
                quad_canvas.paste(quads_r[3], (panel_w, strip_h))

                comp = Image.new("RGB", (ASSIST_MAX_W, strip_h * 3), "white")
                comp.paste(strip_row, (0, 0))
                comp.paste(quad_canvas, (0, strip_h))
                return _save_png_to_data_url(comp)

            # Fallback: just show enhanced full.
            full = _resize_to_width(base, ASSIST_MAX_W)
            return _save_png_to_data_url(full)
    except Exception:
        return None


def maybe_flip_data_url(data_url: str) -> Optional[str]:
    if (not USE_FLIP_VIEW) or (not data_url) or (Image is None):
        return None
    try:
        if "base64," not in data_url:
            return None
        b64 = data_url.split("base64,", 1)[1]
        raw = base64.b64decode(b64)
        with Image.open(io.BytesIO(raw)) as im:
            im = im.convert("RGB")
            flipped = im.transpose(Image.FLIP_LEFT_RIGHT)
            return _save_png_to_data_url(flipped)
    except Exception:
        return None


# ---------------- Question helpers ----------------
def first_line(prompt: str) -> str:
    for ln in prompt.replace("\r\n", "\n").split("\n"):
        ln = ln.strip()
        if ln:
            return ln
    return prompt.strip()


def primitive(question: str) -> str:
    q = question.lower()
    if "equal length" in q or "same length" in q or "length" in q:
        return "length"
    if "straight" in q or "curved" in q or "bent" in q:
        return "straight"
    if "boundary" in q or "border" in q or "adjacent" in q or "adjecent" in q or "regions" in q:
        return "boundary"
    return "other"


# ---------------- Prompts ----------------
def system_text() -> str:
    return """
You are working on a perception-focused image QA benchmark.

Hard constraints (Task I):
- Do NOT do or describe measurement/computation (no pixels, rulers, grids, lengths/angles/distances/areas).
- Use qualitative, human-like visual inspection only.

Critical:
- Images may be PERTURBED / edited. Do NOT rely on illusion-name recall or canonical facts.
- Do NOT mention illusion names (treat them as invalid evidence).

Answering note:
- Questions ask about what is actually drawn in the image (objective geometry / explicit borders), not what it "feels like".
""".strip()


def _checklist(prim: str) -> str:
    if prim == "length":
        return (
            "Checklist (length):\n"
            "- The question asks about TRUE geometric length in the image, not how long it feels under an illusion.\n"
            "- Compare the STRAIGHT horizontal shafts only.\n"
            "- Focus on the two 'corner/junction' points where the slanted fins meet the straight shaft (these define the shaft endpoints).\n"
            "- Inspect BOTH ends: do the left junctions line up between top/bottom? do the right junctions line up?\n"
            "- A clearly visible mismatch of a junction position is strong evidence for NO.\n"
            "- If both junctions align and there is no clear mismatch, that supports YES."
        )
    if prim == "straight":
        return (
            "Checklist (straight):\n"
            "- Inspect each horizontal line for any bowing, kink, or curvature.\n"
            "- Use the stretched/zoomed views to reveal subtle bends.\n"
            "- A single clear bend/kink is strong evidence for NO.\n"
            "- If the edges remain collinear throughout, that supports YES."
        )
    if prim == "boundary":
        return (
            "Checklist (boundary):\n"
            "- Interpret 'boundary' as an explicit separating border/edge line between regions (not merely a color change).\n"
            "- Step 1: Decide whether the image contains MULTIPLE discrete regions (e.g., stripes/blocks) with adjacency.\n"
            "  * A smooth gradient or single uniform field counts as ONE region.\n"
            "- If there is only ONE region (no adjacent regions), the statement is TRUE -> answer YES.\n"
            "- If there are multiple adjacent regions, the statement is universal: EVERY adjacent pair must have an explicit border.\n"
            "- If you can find even ONE adjacency where regions touch/blend with no explicit border line, answer NO."
        )
    return ""


def advocate_prompt(question: str, prim: str, stance: int) -> str:
    side = "YES (answer=1)" if stance == 1 else "NO (answer=0)"
    extra = ""
    if stance == 0:
        extra = (
            "Extra constraint for NO:\n"
            "- You MUST provide at least one concrete, image-locatable counterexample (a specific mismatched endpoint/junction, a kink/bow segment, or an adjacency without an explicit border).\n"
            "- If you cannot find a real counterexample, leave evidence empty and state this in contradictions."
        )
    if prim == "boundary":
        extra = (extra + "\n\n" if extra else "") + (
            "Boundary clarification:\n"
            "- A smooth gradient / single uniform field means there are NO adjacent regions to check -> that supports YES.\n"
            "- For NO, point to a place where two regions meet but there is no explicit separating border line."
        )
    return f"""
You are an advocate for {side}.

Your job:
- Find concrete, image-grounded evidence that supports your side.
- Also list any visible contradictions that weaken your side.

Rules:
- Evidence must be tied to a localized part of the image (e.g., an endpoint, a specific line segment, a specific boundary between two regions).
- No measurement/computation words, no numbers, no pixel/ruler/grid talk.
- Do NOT mention illusion names or canonical facts. If you rely on them, your evidence is invalid.
- If you cannot find real supporting evidence for your side, say so (do not invent).

{extra}

{_checklist(prim)}

Return JSON ONLY (no markdown, no extra text):
{{
  "stance": {stance},
  "evidence": ["...","..."],
  "contradictions": ["..."]
}}

Question:
{question}
""".strip()


def judge_prompt(question: str, prim: str, yes_json: dict, no_json: dict) -> str:
    extra = ""
    if prim == "length":
        extra = (
            "Primitive-specific guidance (length):\n"
            "- Verify claims using the zoom strips/end panels.\n"
            "- Ignore the OUTER arrowhead tips; judge the shaft endpoints at the fin-shaft junction corners.\n"
            "- Choose NO only if you can clearly see at least one junction not lining up between the two lines.\n"
            "- If there is no clearly verifiable junction mismatch, choose YES."
        )
    elif prim == "straight":
        extra = (
            "Primitive-specific guidance (straight):\n"
            "- Use the stretched zoom view to verify whether the thick horizontal line itself bends.\n"
            "- Do not be fooled by radiating background lines; only a visible kink/bow in the thick line counts.\n"
            "- Choose NO only if the bend is clearly verifiable; otherwise choose YES."
        )
    elif prim == "boundary":
        extra = (
            "Primitive-specific guidance (boundary):\n"
            "- Step 1: Decide if there are MULTIPLE discrete regions (e.g., stripes/blocks). A smooth gradient counts as ONE region.\n"
            "- If there is only ONE region (no adjacency), the statement is TRUE -> choose YES.\n"
            "- If there are multiple adjacent regions, 'boundary' means an explicit separating border line.\n"
            "- If adjacent regions touch with no explicit border line anywhere, choose NO."
        )
    return f"""
You are the JUDGE. You can see the image(s) and two advocates' JSON.

Task:
- Decide whether YES (1) or NO (0) is better supported by the image.
- Prefer localized, verifiable visual evidence over generic statements.
- Treat any illusion-name/canonical-fact reasoning as INVALID evidence.
- No measurement/computation.

Decision rule guidance:
- If you can visually confirm a specific mismatch/bend/missing-boundary claimed by the NO advocate, choose NO.
- If NO's counterexample is NOT clearly verifiable in the provided views, do NOT accept it.
- If neither side has verifiable evidence, prefer YES.

{extra}

Primitive: {prim}

YES advocate JSON:
{json.dumps(yes_json, ensure_ascii=False)[:1800]}

NO advocate JSON:
{json.dumps(no_json, ensure_ascii=False)[:1800]}

Return ONLY one line:
<answer>1</answer>  or  <answer>0</answer>

Question:
{question}
""".strip()


def verifier_prompt(question: str, prim: str, yes_json: Optional[dict] = None, no_json: Optional[dict] = None) -> str:
    """
    A primitive-specific checklist that the model must answer in a structured JSON form.
    We then derive the final 0/1 from these qualitative checks (no measurement).
    """
    if prim == "length":
        return f"""
You are the VERIFIER. Inspect the montage views.

Goal: decide whether the two black STRAIGHT shafts are equal length (objective geometry in the image).
Important:
- Ignore OUTER arrow tips. Use the shaft endpoints at the fin-shaft JUNCTION corners.
- Do NOT use illusion-name recall or any measurement.

Decision rule (qualitative):
- Use the TOP strip (first) and BOTTOM strip (second).
- If BOTH ends align vertically (left and right), answer YES.
- If at least one end does NOT align, answer NO.

Return ONLY:
<answer>1</answer> for YES (equal length)
<answer>0</answer> for NO (not equal)

Question:
{question}
""".strip()

    if prim == "straight":
        return f"""
You are the VERIFIER. Inspect the montage views.

Goal: decide whether the two thick horizontal lines are straight (objective geometry in the image).
Important:
- The montage contains multiple candidate bands labeled T1.. and B1.. (top/bottom line candidates).
- Use the red horizontal guideline in each panel as a qualitative reference (no measurement).
- Do NOT be fooled by background radiating lines; only the thick horizontal lines matter.

Checklist:
- Pick top_tag = which T* panel best contains the TOP thick horizontal line.
- Pick bottom_tag = which B* panel best contains the BOTTOM thick horizontal line.
- top_line: is the top thick line straight, not straight, or uncertain?
- bottom_line: same for bottom.

Return JSON ONLY:
{{
  "top_tag": "T1" (or "T2"/"T3"/"T4"/"unknown"),
  "bottom_tag": "B1" (or "B2"/"B3"/"B4"/"unknown"),
  "top_line": "straight" or "not_straight" or "uncertain",
  "bottom_line": "straight" or "not_straight" or "uncertain",
  "best_guess": 0 or 1
}}

Question:
{question}
""".strip()

    if prim == "boundary":
        return f"""
You are the VERIFIER. Inspect the image.

Interpretation:
- Treat "boundary" as an explicit separating border/edge line between adjacent regions (NOT merely a color change).
- A single uniform field or smooth gradient counts as ONE region (no adjacent regions to check).

Checklist:
1) region_type: "single" if the image is one uniform/smooth region; "multiple" if it clearly contains multiple discrete regions (e.g., stripes/blocks).
2) If region_type == "multiple": do you see an explicit border line between EVERY adjacent region pair?

Return JSON ONLY:
{{
  "region_type": "single" or "multiple",
  "explicit_borders_everywhere": "yes" or "no" or "uncertain",
  "best_guess": 0 or 1
}}

Question:
{question}
""".strip()

    # Fallback generic
    return f"""
You are the VERIFIER. Answer the binary question using qualitative visual inspection only.
Return ONLY:
<answer>1</answer> or <answer>0</answer>

Question:
{question}
""".strip()


def build_mm_messages(
    system_txt: str,
    image_path: str,
    assist_data_url: Optional[str],
    assist_flip_data_url: Optional[str],
    user_txt: str,
    include_raw: bool = True,
) -> List[dict]:
    content: List[dict] = []

    if include_raw:
        content.append({"type": "text", "text": "Raw full image (for context):"})
        content.append({"type": "image_url", "image_url": {"url": encode_image_to_data_url(image_path)}})

    if assist_data_url:
        content.append({"type": "text", "text": "Zoom/montage views (use for evidence):"})
        content.append({"type": "image_url", "image_url": {"url": assist_data_url}})
    if assist_flip_data_url:
        content.append({"type": "text", "text": "Zoom/montage views (horizontally flipped; same geometry):"})
        content.append({"type": "image_url", "image_url": {"url": assist_flip_data_url}})

    content.append({"type": "text", "text": user_txt})

    return [
        {"role": "system", "content": system_txt},
        {"role": "user", "content": content},
    ]


# ---------------- Solver ----------------
class MySolver(Solver):
    def __init__(self):
        if not API_KEY:
            print("[WARN] VQA_API_KEY is empty; set it to your DashScope compatible-mode key.")
        # If a model is repeatedly blocked (e.g., free-tier exhausted), stop trying it and use fallback directly.
        self._blocked_models = set()
        self.client = OpenAI(api_key=(API_KEY or None), base_url=(BASE_URL or None))

        self.length_icl: List[Tuple[str, int, str]] = []
        if LENGTH_USE_ICL:
            self.length_icl = build_length_icl_examples()

        print(f"BASE_URL={BASE_URL}")
        print(f"YES_MODEL={YES_MODEL}")
        print(f"NO_MODEL={NO_MODEL}")
        print(f"JUDGE_MODEL={JUDGE_MODEL}")
        print(f"FALLBACK_MODEL={FALLBACK_MODEL}")
        print(f"USE_ASSIST_VIEW={USE_ASSIST_VIEW} USE_FLIP_VIEW={USE_FLIP_VIEW} ASSIST_MAX_W={ASSIST_MAX_W}")
        print(f"LENGTH_USE_ICL={LENGTH_USE_ICL} LENGTH_ICL_PRESET={LENGTH_ICL_PRESET} ICL_EXAMPLES={len(self.length_icl)}")
        if DEBUG_DIR:
            print(f"DEBUG_DIR={DEBUG_DIR}")

    def _call(self, model: str, messages: List[dict], max_tokens: int) -> Optional[str]:
        use_model = model
        if FALLBACK_MODEL and (use_model in self._blocked_models):
            use_model = FALLBACK_MODEL
        try:
            kwargs: Dict[str, Any] = dict(
                model=use_model,
                messages=messages,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=max_tokens,
            )
            if SEED is not None:
                kwargs["seed"] = SEED
            try:
                resp = self.client.chat.completions.create(**kwargs)
            except TypeError:
                kwargs.pop("seed", None)
                resp = self.client.chat.completions.create(**kwargs)

            # Some "compatible" endpoints may return a raw string/HTML.
            if isinstance(resp, str):
                return resp
            if isinstance(resp, dict):
                try:
                    return resp["choices"][0]["message"]["content"]
                except Exception:
                    return None
            return resp.choices[0].message.content if getattr(resp, "choices", None) else None
        except Exception as e:
            err = str(e)
            print(f"[CALL ERROR] model={use_model} err={e}")
            if (
                FALLBACK_MODEL
                and use_model != FALLBACK_MODEL
                and ("AllocationQuota.FreeTierOnly" in err or "Error code: 403" in err or "free tier" in err.lower())
            ):
                self._blocked_models.add(use_model)
                try:
                    print(f"[CALL RETRY] falling back to model={FALLBACK_MODEL}")
                    kwargs: Dict[str, Any] = dict(
                        model=FALLBACK_MODEL,
                        messages=messages,
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                        max_tokens=max_tokens,
                    )
                    if SEED is not None:
                        kwargs["seed"] = SEED
                    try:
                        resp = self.client.chat.completions.create(**kwargs)
                    except TypeError:
                        kwargs.pop("seed", None)
                        resp = self.client.chat.completions.create(**kwargs)
                    if isinstance(resp, str):
                        return resp
                    if isinstance(resp, dict):
                        try:
                            return resp["choices"][0]["message"]["content"]
                        except Exception:
                            return None
                    return resp.choices[0].message.content if getattr(resp, "choices", None) else None
                except Exception as e2:
                    print(f"[CALL ERROR] model={FALLBACK_MODEL} err={e2}")
            return None

    def solve(self, image_path: str, prompt: str) -> int:
        # Robust to different working dirs.
        if image_path and (not os.path.isabs(image_path)) and (not os.path.exists(image_path)):
            alt = os.path.join(_THIS_DIR, image_path)
            if os.path.exists(alt):
                image_path = alt

        q = first_line(prompt)
        prim = primitive(q)
        sys_txt = system_text()

        assist = build_assist_montage(image_path, prim=prim)
        assist_flip = maybe_flip_data_url(assist) if assist else None
        include_raw = (prim == "boundary") or (assist is None)

        dbg = _ensure_debug_dir()
        if dbg and assist and "base64," in assist:
            try:
                raw = base64.b64decode(assist.split("base64,", 1)[1])
                stem = os.path.splitext(os.path.basename(image_path))[0]
                _debug_write(os.path.join(dbg, f"{stem}_{prim}_montage.png"), raw)
            except Exception:
                pass

        # Single-call verifier (v9 default): no advocates to avoid "illusion prior" priming.
        if prim == "length" and self.length_icl:
            # Length-only ICL: show a few synthetic reference examples to counter the illusion prior.
            content: List[dict] = []
            content.append({"type": "text", "text": "Reference examples:"})
            for name, ans_ex, url in self.length_icl:
                content.append({"type": "text", "text": f"{name} (label={ans_ex}):"})
                content.append({"type": "image_url", "image_url": {"url": url}})
                content.append({"type": "text", "text": f"<answer>{int(ans_ex)}</answer>"})

            content.append({"type": "text", "text": "Query montage (answer unknown):"})
            if assist:
                content.append({"type": "image_url", "image_url": {"url": assist}})
            else:
                # Fallback if montage failed.
                content.append({"type": "image_url", "image_url": {"url": encode_image_to_data_url(image_path)}})
            if assist_flip:
                content.append({"type": "text", "text": "Same montage, horizontally flipped (geometry unchanged):"})
                content.append({"type": "image_url", "image_url": {"url": assist_flip}})

            content.append({"type": "text", "text": verifier_prompt(q, prim)})

            ver_msg = [
                {"role": "system", "content": sys_txt},
                {"role": "user", "content": content},
            ]
        else:
            ver_msg = build_mm_messages(
                sys_txt,
                image_path,
                assist,
                assist_flip,
                verifier_prompt(q, prim),
                include_raw=(prim == "boundary") or (assist is None),
            )
        ver_out = self._call(JUDGE_MODEL, ver_msg, max_tokens=max(JUDGE_MAX_TOKENS, 240)) or ""
        ver_obj = _extract_json(ver_out)
        ans = derive_answer_from_verifier(prim, ver_obj)

        if dbg:
            stem = os.path.splitext(os.path.basename(image_path))[0]
            _debug_write_text(os.path.join(dbg, f"{stem}_{prim}_verifier.txt"), ver_out)

        if ans in (0, 1):
            return ans

        # If the model used <answer> tags (e.g., length prompt), accept it without a retry.
        a_tag = parse_answer(ver_out)
        if a_tag in (0, 1):
            return a_tag

        # One cheap format-repair retry if the model didn't follow the requested format.
        retry = "Follow the required output format exactly. Return ONLY the requested JSON or <answer> tags.\n\n" + verifier_prompt(q, prim)
        ver_msg2 = build_mm_messages(
            sys_txt,
            image_path,
            assist,
            assist_flip,
            retry,
            include_raw=(prim == "boundary") or (assist is None),
        )
        ver_out2 = self._call(JUDGE_MODEL, ver_msg2, max_tokens=max(JUDGE_MAX_TOKENS, 240)) or ""
        ver_obj2 = _extract_json(ver_out2)
        ans2 = derive_answer_from_verifier(prim, ver_obj2)

        if dbg:
            stem = os.path.splitext(os.path.basename(image_path))[0]
            _debug_write_text(os.path.join(dbg, f"{stem}_{prim}_verifier_retry.txt"), ver_out2)

        if ans2 in (0, 1):
            return ans2

        # Fallback: allow <answer> tags if JSON parse failed.
        a = parse_answer(ver_out2 or ver_out)
        return a if a in (0, 1) else 0

    def model_info(self) -> dict:
        return {
            "model": JUDGE_MODEL,
            "parameters": {
                "base_url": BASE_URL,
                "yes_model": YES_MODEL,
                "no_model": NO_MODEL,
                "judge_model": JUDGE_MODEL,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "seed": SEED,
                "use_assist_view": USE_ASSIST_VIEW,
                "use_flip_view": USE_FLIP_VIEW,
                "assist_max_w": ASSIST_MAX_W,
                "assist_contrast": ASSIST_CONTRAST,
                "assist_sharpness": ASSIST_SHARPNESS,
                "boundary_contrast": BOUNDARY_CONTRAST,
                "boundary_sharpness": BOUNDARY_SHARPNESS,
                "length_strip_h": LENGTH_STRIP_H,
                "length_use_icl": LENGTH_USE_ICL,
                "length_icl_preset": LENGTH_ICL_PRESET,
                "boundary_strip_half_w": BOUNDARY_STRIP_HALF_W,
                "straight_cand_top": STRAIGHT_CAND_TOP,
                "straight_cand_bottom": STRAIGHT_CAND_BOTTOM,
                "straight_cand_half_h": STRAIGHT_CAND_HALF_H,
                "straight_cand_panel_h": STRAIGHT_CAND_PANEL_H,
                "pipeline": "single-call verifier: length uses montage + synthetic ICL (if enabled), straight uses multi-candidate bands+guideline, boundary raw-only -> verifier checklist (JSON or <answer>) -> derived 0/1; no measurement",
            },
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task1 verifier v9 runner (alternative)")
    parser.add_argument("--input-csv", default="val.csv", help="Input CSV file path")
    parser.add_argument("--output-txt", default="predictions_debate_v9.txt", help="Output TXT file path")
    parser.add_argument("--output-json", default="model_debate_v9.json", help="Output JSON file path")
    args = parser.parse_args()
    run(MySolver(), args.input_csv, args.output_txt, args.output_json)
