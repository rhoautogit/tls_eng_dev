"""
Reconstruct a PDF purely from rich extracted data (no access to the source PDF).

Reads output/<stem>/rich/page_NNN_visual.json files and rebuilds each page:
  1. Vector drawings (backgrounds, borders, lines, shapes)
  2. Embedded images at their display positions
  3. Text spans with correct font, size, color, and position

Usage:
    python reconstruct_pdf.py                            # auto-detect from output/
    python reconstruct_pdf.py <pdf_stem>                 # e.g. "Test"
"""

import base64
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF


# ── Font mapping ─────────────────────────────────────────────────────────────
# Map common PDF font names to PyMuPDF built-in (Base14) equivalents.
# PyMuPDF can insert text with these without needing external font files.

FONT_MAP = {
    # Helvetica family
    "arial": "helv",
    "arialmt": "helv",
    "arial-boldmt": "hebo",
    "arial-italicmt": "heit",
    "arial-bolditalicmt": "hebi",
    "helvetica": "helv",
    "helvetica-bold": "hebo",
    "helvetica-oblique": "heit",
    "helvetica-boldoblique": "hebi",
    # Times family
    "times": "tiro",
    "times-roman": "tiro",
    "times-bold": "tibo",
    "times-italic": "tiit",
    "times-bolditalic": "tibi",
    "timesnewroman": "tiro",
    "timesnewromanpsmt": "tiro",
    "timesnewromanps-boldmt": "tibo",
    "timesnewromanps-italicmt": "tiit",
    "timesnewromanps-bolditalicmt": "tibi",
    # Courier family
    "courier": "cour",
    "courier-bold": "cobo",
    "courier-oblique": "coit",
    "courier-boldoblique": "cobi",
    "couriernew": "cour",
    "couriernewpsmt": "cour",
    # Symbol / Zapf
    "symbol": "symb",
    "zapfdingbats": "zadb",
}


def _map_font(name: str) -> str:
    """Map a PDF font name to a PyMuPDF built-in fontname."""
    key = name.lower().replace(" ", "").replace("-", "").replace(",", "")
    # Direct match
    if key in FONT_MAP:
        return FONT_MAP[key]
    # Try stripping common suffixes
    for suffix in ("mt", "psmt", "ps", "ms"):
        stripped = key.rstrip(suffix) if key.endswith(suffix) else key
        if stripped in FONT_MAP:
            return FONT_MAP[stripped]
    # Heuristic: check if name contains a known family
    lower = name.lower()
    if "bold" in lower and "italic" in lower:
        if "times" in lower:
            return "tibi"
        if "courier" in lower:
            return "cobi"
        return "hebi"
    if "bold" in lower:
        if "times" in lower:
            return "tibo"
        if "courier" in lower:
            return "cobo"
        return "hebo"
    if "italic" in lower or "oblique" in lower:
        if "times" in lower:
            return "tiit"
        if "courier" in lower:
            return "coit"
        return "heit"
    if "times" in lower:
        return "tiro"
    if "courier" in lower:
        return "cour"
    # Default fallback
    return "helv"


def _hex_to_rgb(hex_str: str) -> Tuple[float, float, float]:
    """Convert '#RRGGBB' to (r, g, b) in 0-1 range."""
    h = hex_str.lstrip("#")
    if len(h) != 6:
        return (0.0, 0.0, 0.0)
    return (int(h[0:2], 16) / 255.0,
            int(h[2:4], 16) / 255.0,
            int(h[4:6], 16) / 255.0)


# ── Drawing reconstruction ───────────────────────────────────────────────────

def _style_key(d: Dict[str, Any]) -> tuple:
    """Create a hashable key from drawing style properties for batching."""
    color = tuple(d["color"]) if d.get("color") else None
    fill = tuple(d["fill"]) if d.get("fill") else None
    return (color, fill, d.get("width", 0.0), d.get("closePath", False),
            d.get("even_odd", False), d.get("fill_opacity", 1.0),
            d.get("stroke_opacity", 1.0))


def _draw_items_to_shape(shape: fitz.Shape, items: List[Dict]):
    """Add drawing items to an existing Shape object."""
    for item in items:
        kind = item["type"]
        try:
            if kind == "l":
                shape.draw_line(fitz.Point(item["p1"]), fitz.Point(item["p2"]))
            elif kind == "re":
                shape.draw_rect(fitz.Rect(item["rect"]))
            elif kind == "c":
                shape.draw_bezier(
                    fitz.Point(item["p1"]), fitz.Point(item["p2"]),
                    fitz.Point(item["p3"]), fitz.Point(item["p4"]),
                )
            elif kind == "qu":
                shape.draw_quad(fitz.Quad(
                    fitz.Point(item["ul"]), fitz.Point(item["ur"]),
                    fitz.Point(item["ll"]), fitz.Point(item["lr"]),
                ))
        except Exception:
            continue


def _draw_all_paths(page: fitz.Page, drawings: List[Dict[str, Any]],
                    ctm_scale: float = 1.0):
    """Draw all vector paths on a page, batching by visual style for speed.

    Uses the CTM scale factor from the rich extraction data to correct
    stroke widths that PyMuPDF's get_drawings() reports without applying
    the Form XObject's coordinate transform.
    """
    if not drawings:
        return

    # Group drawings by style so we can batch them into fewer shape.finish() calls
    from collections import defaultdict
    batches: dict[tuple, list] = defaultdict(list)
    for d in drawings:
        key = _style_key(d)
        batches[key].append(d)

    for style_key, batch in batches.items():
        color, fill, width, close, even_odd, f_op, s_op = style_key
        shape = page.new_shape()

        # Apply CTM scale to correct widths from Form XObjects;
        # keep fill-only paths (width=0) untouched
        draw_width = width * ctm_scale if width > 0 else 0

        for d in batch:
            items = d.get("items", [])
            if items:
                _draw_items_to_shape(shape, items)
                # finish each sub-path individually to preserve closePath per drawing
                try:
                    shape.finish(
                        color=list(color) if color else None,
                        fill=list(fill) if fill else None,
                        width=draw_width,
                        closePath=close,
                        even_odd=even_odd,
                        fill_opacity=f_op,
                        stroke_opacity=s_op,
                    )
                except Exception:
                    pass

        try:
            shape.commit()
        except Exception:
            pass


# ── Checkbox drawing ─────────────────────────────────────────────────────────

# Unicode markers used by the OCR rich extractor for checkboxes
_CHECKBOX_CHECKED = "\u2611"   # ☑
_CHECKBOX_UNCHECKED = "\u2610" # ☐
_CHECKBOX_CHARS = {_CHECKBOX_CHECKED, _CHECKBOX_UNCHECKED}


def _draw_checkbox(page: fitz.Page, origin: List[float], size: float,
                   checked: bool):
    """Draw a checkbox at the given origin point.

    Draws a square box and, if checked, an X mark inside it.
    The box size is based on the font size of the surrounding text.
    """
    box_size = size * 0.85
    # Origin is baseline-left; adjust to get the top-left of the box
    x0 = origin[0]
    y0 = origin[1] - box_size * 0.85  # shift up from baseline
    x1 = x0 + box_size
    y1 = y0 + box_size

    rect = fitz.Rect(x0, y0, x1, y1)
    shape = page.new_shape()

    # Draw the box outline
    shape.draw_rect(rect)
    shape.finish(color=(0, 0, 0), fill=None, width=0.8)

    if checked:
        # Draw X mark inside the box
        margin = box_size * 0.15
        shape.draw_line(
            fitz.Point(x0 + margin, y0 + margin),
            fitz.Point(x1 - margin, y1 - margin),
        )
        shape.finish(color=(0, 0, 0), width=1.0)
        shape.draw_line(
            fitz.Point(x1 - margin, y0 + margin),
            fitz.Point(x0 + margin, y1 - margin),
        )
        shape.finish(color=(0, 0, 0), width=1.0)

    shape.commit()


# ── Page reconstruction ──────────────────────────────────────────────────────

def reconstruct_page(page: fitz.Page, data: Dict[str, Any]):
    """Reconstruct a single page from its rich visual data."""

    # ── 1. Vector drawings (backgrounds, borders, shapes) ────────────────
    ctm_scale = data.get("ctm_scale", 1.0)
    drawings = sorted(data.get("drawings", []), key=lambda d: d.get("seqno", 0))
    _draw_all_paths(page, drawings, ctm_scale=ctm_scale)

    # ── 2. Images ────────────────────────────────────────────────────────
    for img in data.get("images", []):
        bbox = img.get("bbox")
        b64 = img.get("data_b64", "")
        if not bbox or not b64:
            continue
        try:
            img_bytes = base64.b64decode(b64)
            rect = fitz.Rect(bbox)
            page.insert_image(rect, stream=img_bytes, keep_proportion=True)
        except Exception as exc:
            pass  # skip undecodable images

    # ── 3. Text spans (with checkbox support) ─────────────────────────────
    for block in data.get("text_blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "")
                if not text:
                    continue
                origin = span.get("origin", [0, 0])
                font_size = span.get("size", 10.0)

                # Handle checkbox unicode markers
                if text in _CHECKBOX_CHARS:
                    checked = (text == _CHECKBOX_CHECKED)
                    _draw_checkbox(page, origin, font_size, checked)
                    continue

                font_name = _map_font(span.get("font", "Helvetica"))
                color = _hex_to_rgb(span.get("color", "#000000"))

                try:
                    page.insert_text(
                        fitz.Point(origin),
                        text,
                        fontname=font_name,
                        fontsize=font_size,
                        color=color,
                    )
                except Exception:
                    # Fallback: try with default font
                    try:
                        page.insert_text(
                            fitz.Point(origin),
                            text,
                            fontname="helv",
                            fontsize=font_size,
                            color=color,
                        )
                    except Exception:
                        pass

    # ── 4. Form elements (standalone, not from text spans) ────────────────
    for elem in data.get("form_elements", []):
        if elem.get("type") == "checkbox":
            bbox = elem.get("bbox", [0, 0, 0, 0])
            checked = elem.get("checked", False)
            # Use bbox center as origin, size from bbox height
            box_size = bbox[3] - bbox[1]
            origin = [bbox[0], bbox[3] - box_size * 0.15]
            _draw_checkbox(page, origin, box_size, checked)


# ── Main reconstruction ──────────────────────────────────────────────────────

def find_rich_dir(script_dir: Path, argv: list) -> Path:
    """Locate the rich/ directory for the PDF to reconstruct."""
    output_base = script_dir / "output"

    if len(argv) > 1:
        stem = argv[1]
        candidate = output_base / stem / "rich"
        if candidate.exists():
            return candidate

    # Auto-detect: find the first output dir that has a rich/ subfolder
    if output_base.exists():
        for d in sorted(output_base.iterdir()):
            rich = d / "rich"
            if rich.exists() and any(rich.glob("page_*_visual.json")):
                return rich

    raise FileNotFoundError(
        "No rich extraction data found. Run the pipeline first, then reconstruct.\n"
        "Usage: python reconstruct_pdf.py [pdf_stem]"
    )


def reconstruct(rich_dir: Path, output_path: Path):
    json_files = sorted(rich_dir.glob("page_*_visual.json"))
    if not json_files:
        print(f"No visual JSON files found in {rich_dir}")
        sys.exit(1)

    print(f"Rich data  : {rich_dir}")
    print(f"Pages found: {len(json_files)}")
    print(f"Output     : {output_path}")

    out_doc = fitz.open()

    for i, json_file in enumerate(json_files):
        data = json.loads(json_file.read_text(encoding="utf-8"))
        width = data.get("width", 612.0)
        height = data.get("height", 792.0)

        page = out_doc.new_page(width=width, height=height)
        reconstruct_page(page, data)

        print(f"  Page {i + 1:>3}/{len(json_files)} done")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_doc.save(str(output_path), garbage=4, deflate=True)
    out_doc.close()

    size_kb = output_path.stat().st_size / 1024
    print(f"\nSaved -> {output_path}  ({size_kb:.1f} KB)")
    return output_path


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    script_dir = Path(__file__).parent

    rich_dir = find_rich_dir(script_dir, sys.argv)
    output_dir = script_dir / "Test Outputs"
    output_name = "TEST_OUTPUT_TLS"

    reconstruct(rich_dir, output_dir / f"{output_name}.pdf")
