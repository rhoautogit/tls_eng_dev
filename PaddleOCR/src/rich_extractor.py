"""Rich visual extraction using PyMuPDF's detailed APIs.

Captures the full visual state of each PDF page:
  - Text spans with font, size, color, bold/italic, exact (x,y) origin
  - Vector drawings (lines, rects, curves) with stroke/fill colors and opacity
  - Embedded images as base64-encoded bytes with display rects

Output is a single JSON file per page that contains everything needed to
reconstruct the page without access to the original PDF.
"""
from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import re

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


# ── Serialisation helpers ────────────────────────────────────────────────────

def _pt(p) -> List[float]:
    """fitz.Point / tuple -> [x, y]"""
    return [round(float(p[0]), 4), round(float(p[1]), 4)]


def _rc(r) -> List[float]:
    """fitz.Rect / tuple -> [x0, y0, x1, y1]"""
    return [round(float(r[0]), 4), round(float(r[1]), 4),
            round(float(r[2]), 4), round(float(r[3]), 4)]


def _color_int_to_hex(c: int) -> str:
    """Convert PyMuPDF's integer sRGB (e.g. 0x000000) to '#RRGGBB'."""
    if c is None:
        return "#000000"
    return f"#{c & 0xFFFFFF:06X}"


def _serialize_color(c) -> Optional[List[float]]:
    """Convert a PyMuPDF color tuple (r, g, b) in 0-1 range to a list, or None."""
    if c is None:
        return None
    return [round(float(x), 4) for x in c]


def _serialize_drawing_items(items: list) -> List[Dict[str, Any]]:
    """Convert PyMuPDF drawing-item tuples to JSON-friendly dicts."""
    result = []
    for item in items:
        kind = item[0]
        if kind == "l":  # line: ("l", Point, Point)
            result.append({"type": "l", "p1": _pt(item[1]), "p2": _pt(item[2])})
        elif kind == "re":  # rectangle: ("re", Rect)
            result.append({"type": "re", "rect": _rc(item[1])})
        elif kind == "c":  # cubic Bezier: ("c", P1, P2, P3, P4)
            result.append({
                "type": "c",
                "p1": _pt(item[1]), "p2": _pt(item[2]),
                "p3": _pt(item[3]), "p4": _pt(item[4]),
            })
        elif kind == "qu":  # quad: ("qu", Quad)
            q = item[1]
            result.append({
                "type": "qu",
                "ul": _pt(q.ul), "ur": _pt(q.ur),
                "ll": _pt(q.ll), "lr": _pt(q.lr),
            })
        else:
            logger.debug("Unknown drawing item kind: %s", kind)
    return result


# ── CTM detection ────────────────────────────────────────────────────────────

def _detect_ctm_scale(doc: fitz.Document, page: fitz.Page) -> float:
    """Detect the minimum scaling CTM from Form XObjects on this page.

    PyMuPDF's ``get_drawings()`` transforms coordinates to page space but
    does NOT scale stroke widths by the CTM.  Engineering PDFs commonly
    wrap drawings in a Form XObject with e.g. ``0.1 0 0 0.1 0 0 cm`` so
    that coordinates are 10x and widths appear 10x bolder unless corrected.

    Returns the smallest uniform scale factor found (< 1.0), or 1.0 if
    no scaling XObjects exist.
    """
    min_scale = 1.0
    try:
        for xo in page.get_xobjects():
            xref = xo[0]
            try:
                stream = doc.xref_stream(xref).decode("latin-1")
                # Look for the first CTM operator in the stream
                match = re.search(
                    r"([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+"
                    r"([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+cm",
                    stream,
                )
                if match:
                    a = float(match.group(1))
                    d = float(match.group(4))
                    # Uniform scale with magnitude < 1 means widths are inflated
                    if abs(a) < 1.0 and abs(d) < 1.0 and abs(a - d) < 0.01:
                        min_scale = min(min_scale, abs(a))
            except Exception:
                continue
    except Exception:
        pass
    return round(min_scale, 6)


# ── Core extraction ──────────────────────────────────────────────────────────

def extract_rich_page(pdf_path: str, page_num: int) -> Dict[str, Any]:
    """Extract the complete visual state of a single page.

    Returns a dict that can be serialised to JSON and later used to
    reconstruct the page without accessing the original PDF.
    """
    doc = fitz.open(pdf_path)
    page = doc[page_num]

    page_width = page.rect.width
    page_height = page.rect.height

    # ── Text ─────────────────────────────────────────────────────────────
    text_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
    text_blocks = []
    fonts_used: set[str] = set()

    for block in text_dict.get("blocks", []):
        if block["type"] != 0:  # skip image blocks (handled separately)
            continue
        lines_out = []
        for line in block.get("lines", []):
            spans_out = []
            for span in line.get("spans", []):
                fonts_used.add(span["font"])
                spans_out.append({
                    "text": span["text"],
                    "font": span["font"],
                    "size": round(span["size"], 2),
                    "flags": span["flags"],
                    "color": _color_int_to_hex(span["color"]),
                    "origin": _pt(span["origin"]),
                    "bbox": _rc(span["bbox"]),
                    "ascender": round(span.get("ascender", 0.0), 4),
                    "descender": round(span.get("descender", 0.0), 4),
                })
            lines_out.append({
                "bbox": _rc(line["bbox"]),
                "wmode": line.get("wmode", 0),
                "dir": _pt(line.get("dir", (1.0, 0.0))),
                "spans": spans_out,
            })
        text_blocks.append({
            "bbox": _rc(block["bbox"]),
            "lines": lines_out,
        })

    # ── Drawings (vector paths) ──────────────────────────────────────────
    drawings_out = []
    for d in page.get_drawings():
        w = d.get("width")
        lc = d.get("lineCap")
        lj = d.get("lineJoin")
        fo = d.get("fill_opacity")
        so = d.get("stroke_opacity")
        da = d.get("dashes")
        drawings_out.append({
            "seqno": d.get("seqno", 0),
            "items": _serialize_drawing_items(d.get("items", [])),
            "color": _serialize_color(d.get("color")),
            "fill": _serialize_color(d.get("fill")),
            "width": round(float(w), 4) if w is not None else 0.0,
            "lineCap": list(lc) if lc is not None else [0, 0, 0],
            "lineJoin": int(lj) if lj is not None else 0,
            "dashes": da if da is not None else "",
            "closePath": bool(d.get("closePath", False)),
            "even_odd": bool(d.get("even_odd", False)),
            "fill_opacity": round(float(fo), 4) if fo is not None else 1.0,
            "stroke_opacity": round(float(so), 4) if so is not None else 1.0,
            "rect": _rc(d.get("rect", (0, 0, 0, 0))),
        })

    # ── Images ───────────────────────────────────────────────────────────
    images_out = []
    for img_info in page.get_images(full=True):
        xref = img_info[0]
        try:
            img_data = doc.extract_image(xref)
            img_bytes = img_data["image"]
            ext = img_data.get("ext", "png")

            # Get display position(s) on the page
            rects = page.get_image_rects(xref)
            display_rect = _rc(rects[0]) if rects else None

            images_out.append({
                "bbox": display_rect,
                "width": img_data.get("width", 0),
                "height": img_data.get("height", 0),
                "ext": ext,
                "data_b64": base64.b64encode(img_bytes).decode("ascii"),
                "xref": xref,
            })
        except Exception as exc:
            logger.debug("Could not extract image xref=%d p%d: %s", xref, page_num, exc)

    # ── CTM scale factor ────────────────────────────────────────────────
    # Form XObjects may apply a scaling CTM (e.g. 0.1 0 0 0.1 0 0 cm).
    # PyMuPDF's get_drawings() transforms coordinates to page space but
    # does NOT scale stroke widths by the CTM.  We detect the scale here
    # so the reconstruction can apply it to line widths.
    ctm_scale = _detect_ctm_scale(doc, page)

    doc.close()

    return {
        "version": "1.1",
        "page_num": page_num,
        "width": round(page_width, 4),
        "height": round(page_height, 4),
        "ctm_scale": ctm_scale,
        "text_blocks": text_blocks,
        "drawings": drawings_out,
        "images": images_out,
        "fonts_used": sorted(fonts_used),
    }


# ── Persistence ──────────────────────────────────────────────────────────────

def save_rich_page(data: Dict[str, Any], rich_dir: Path, page_num: int) -> Path:
    """Write the rich visual data to a JSON file."""
    rich_dir.mkdir(parents=True, exist_ok=True)
    page_label = f"page_{page_num + 1:03d}"
    out_path = rich_dir / f"{page_label}_visual.json"
    out_path.write_text(
        json.dumps(data, ensure_ascii=False),
        encoding="utf-8",
    )
    return out_path
