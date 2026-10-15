"""From-scratch, dependency-light OCR for these specific maps.

No Tesseract / EasyOCR is required. Every map on this dataset is rendered by
the same matplotlib script in the same font (DejaVu Sans, bundled with
matplotlib), so two cheap, robust strategies cover everything we need to
read:

1. ``DigitReader`` -- per-character template matching restricted to the
   small closed charset ``0-9 . - :`` that appears in tick labels, bubble
   values and dates. A tiny, closed charset makes per-glyph classification
   reliable even on JPEG-compressed anti-aliased text.

2. ``VocabularyReader`` -- whole-word/whole-phrase template matching against
   a *finite* list of known strings (station names, "Minimum"/"Maximum").
   Segmenting and classifying individual letters of full text is unnecessary
   (and less robust) when the space of possible outputs is small and known
   in advance -- we just compare the crop, as one image, against a rendered
   reference of each candidate string and keep the best match.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage import measure

from .config import DIGIT_CHARSET, FONT_REGULAR


def _render_char_mask(ch: str, font_path: str, size: int) -> np.ndarray | None:
    font = ImageFont.truetype(font_path, size)
    canvas = Image.new("L", (size * 3, size * 3), color=255)
    d = ImageDraw.Draw(canvas)
    d.text((size, size // 2), ch, font=font, fill=0)
    arr = np.array(canvas)
    mask = arr < 128
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _render_text_mask(text: str, font_path: str, size: int) -> np.ndarray:
    font = ImageFont.truetype(font_path, size)
    tmp = Image.new("L", (10, 10), 255)
    bbox = ImageDraw.Draw(tmp).textbbox((0, 0), text, font=font)
    w = max(1, bbox[2] - bbox[0]) + 8
    h = max(1, bbox[3] - bbox[1]) + 8
    canvas = Image.new("L", (w, h), color=255)
    ImageDraw.Draw(canvas).text((4 - bbox[0], 4 - bbox[1]), text, font=font, fill=0)
    arr = np.array(canvas)
    mask = arr < 128
    if not mask.any():
        return mask
    ys, xs = np.where(mask)
    return mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _normalize(mask: np.ndarray, out: tuple[int, int]) -> np.ndarray:
    """Resize a boolean glyph mask to a fixed-size float template."""
    im = Image.fromarray((mask * 255).astype(np.uint8))
    im = im.resize((out[1], out[0]), Image.LANCZOS)
    return np.asarray(im, dtype=np.float32) / 255.0


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized cross-correlation between two same-shape float arrays."""
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-6:
        return -1.0
    return float((a * b).sum() / denom)


def dark_mask_from_gray(gray_crop: np.ndarray, thresh: int = 150) -> np.ndarray:
    """Text mask for glyphs sitting on a plain white background (titles,
    axis tick labels, annotation text)."""
    return gray_crop < thresh


def dark_mask_from_rgb(rgb_crop: np.ndarray, thresh: int = 60) -> np.ndarray:
    """Text mask for glyphs that may sit on a *coloured* background (bubble
    values). Bubble text is rendered near-pure black; every fill colour in
    the coolwarm colormap used for these bubbles keeps at least one channel
    well above black, so requiring all three channels to be dark isolates
    the text/ring strokes regardless of how dark the fill itself is (e.g.
    the near-navy fill used for the coldest temperatures)."""
    r, g, b = rgb_crop[..., 0], rgb_crop[..., 1], rgb_crop[..., 2]
    return (r < thresh) & (g < thresh) & (b < thresh)


def _segment_components(dark: np.ndarray, min_area: int = 4):
    lbl = measure.label(dark, connectivity=2)
    comps = []
    for r in measure.regionprops(lbl):
        if r.area < min_area:
            continue
        y0, x0, y1, x1 = r.bbox
        comps.append((x0, y0, x1, y1))
    comps.sort(key=lambda c: c[0])
    return comps


@dataclass
class Token:
    char: str
    score: float
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1


class DigitReader:
    """Template-matching reader for the closed charset ``0-9 . - :``."""

    def __init__(self, font_path: str = FONT_REGULAR, template_size: tuple[int, int] = (32, 24),
                 render_px: int = 64):
        self.template_size = template_size
        self.templates: dict[str, np.ndarray] = {}
        for ch in DIGIT_CHARSET:
            if ch == ".":
                continue  # too small/inconsistent to template-match; handled
                          # geometrically in read_tokens() by component size
            mask = _render_char_mask(ch, font_path, render_px)
            if mask is not None:
                self.templates[ch] = _normalize(mask, template_size)

    def read_tokens(self, dark: np.ndarray, min_area: int = 4, min_score: float = 0.5,
                     border_margin: int = 1) -> list[Token]:
        """``dark`` is a pre-built boolean text mask -- see
        :func:`dark_mask_from_gray` / :func:`dark_mask_from_rgb`."""
        h, w = dark.shape
        comps = _segment_components(dark, min_area)
        comps = [c for c in comps
                 if c[0] > border_margin and c[1] > border_margin
                 and c[2] < w - border_margin and c[3] < h - border_margin]
        if not comps:
            return []

        # Two passes: first find the "real" (digit-sized) components so a
        # small candidate's height, and the baseline they sit on, are known
        # *before* deciding whether any small component is a decimal point.
        # A digit like "3" can fragment into a tiny extra blob near its own
        # top on compression noise; a genuine "." always sits at the row's
        # baseline, not wherever a digit happened to fragment -- so a small
        # blob only counts as "." if it's actually down near that baseline.
        # The reference height is the *median* (not max) so a single huge
        # outlier component (e.g. a bubble's ring, if one leaks into the
        # crop) can't dominate it the way a single tiny fragment can't
        # either -- digits are the majority of components in practice.
        all_heights = [c[3] - c[1] for c in comps]
        rough_med_h = float(np.median(all_heights))
        digit_sized = [c for c in comps
                        if 0.5 * rough_med_h <= (c[3] - c[1]) <= 1.8 * rough_med_h]
        if not digit_sized:
            digit_sized = comps
        heights = [c[3] - c[1] for c in digit_sized]
        med_h = float(np.median(heights))
        baseline = float(np.median([c[3] for c in digit_sized]))

        tokens: list[Token] = []
        for (x0, y0, x1, y1) in comps:
            bw, bh = x1 - x0, y1 - y0
            if bh < 0.35 * med_h and bw < 0.35 * med_h:
                if abs(y1 - baseline) <= 0.3 * med_h:
                    tokens.append(Token(".", 1.0, (x0, y0, x1, y1)))
                continue  # small and off-baseline -> fragment noise, drop
            if bh < 0.3 * med_h or bh > 1.6 * med_h:
                continue  # implausible size for this row -> noise
            crop = dark[y0:y1, x0:x1]
            norm = _normalize(crop, self.template_size)
            best_ch, best_s = None, -2.0
            for ch, tmpl in self.templates.items():
                s = _ncc(norm, tmpl)
                if s > best_s:
                    best_s, best_ch = s, ch
            if best_s < min_score:
                continue
            tokens.append(Token(best_ch, best_s, (x0, y0, x1, y1)))
        tokens.sort(key=lambda t: t.bbox[0])
        return tokens

    def read_string(self, dark: np.ndarray, **kwargs) -> str:
        return "".join(t.char for t in self.read_tokens(dark, **kwargs))

    def read_float(self, dark: np.ndarray, **kwargs) -> float | None:
        s = self.read_string(dark, **kwargs)
        s = s.strip(":")
        try:
            return float(s)
        except ValueError:
            return None


def _dominant_row_band(dark: np.ndarray, gap_rows: int = 2) -> np.ndarray:
    """Keep only the horizontal band of rows with the most dark pixels.

    Station-label crops are generous boxes that can still catch a sliver of
    the bubble's ring or the coastline outline; those are thin arcs, while
    an actual text line is a dense, multi-row band, so picking the
    highest-total-density contiguous row run isolates the text reliably.
    """
    row_counts = dark.sum(axis=1)
    nonzero = row_counts > 0
    if not nonzero.any():
        return dark
    bands = []
    start = None
    gap = 0
    for i, v in enumerate(nonzero):
        if v:
            if start is None:
                start = i
            gap = 0
        elif start is not None:
            gap += 1
            if gap > gap_rows:
                bands.append((start, i - gap))
                start = None
                gap = 0
    if start is not None:
        bands.append((start, len(nonzero) - 1))
    if not bands:
        return dark
    best = max(bands, key=lambda b: row_counts[b[0]:b[1] + 1].sum())
    out = np.zeros_like(dark)
    out[best[0]:best[1] + 1, :] = dark[best[0]:best[1] + 1, :]
    return out


class VocabularyReader:
    """Whole-string template matcher over a fixed, known vocabulary."""

    def __init__(self, vocabulary: list[str], font_path: str = FONT_REGULAR,
                 font_size: int = 40, canvas: tuple[int, int] = (48, 420)):
        self.canvas = canvas
        self.templates: dict[str, np.ndarray] = {}
        for word in vocabulary:
            mask = _render_text_mask(word, font_path, font_size)
            self.templates[word] = _normalize(mask, canvas)

    def match(self, gray_crop: np.ndarray, dark_thresh: int = 150) -> tuple[str, float]:
        dark = gray_crop < dark_thresh
        if not dark.any():
            return "", -1.0
        dark = _dominant_row_band(dark)
        if not dark.any():
            return "", -1.0
        ys, xs = np.where(dark)
        crop = dark[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        norm = _normalize(crop, self.canvas)
        best_word, best_s = "", -2.0
        for word, tmpl in self.templates.items():
            s = _ncc(norm, tmpl)
            if s > best_s:
                best_s, best_word = s, word
        return best_word, best_s
