from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
from typing import Any

from PIL import Image, ImageChops, ImageDraw, ImageStat

from .atomic import atomic_write_json, atomic_write_text, sha256_file
from .history_media import verify_sidecar


FIGURE_IDS = (
    "COMPANION_V02_FIGURE_03_DENSITY_FIRST_PASSAGE",
    "COMPANION_V02_FIGURE_04_CONTROLS_AND_ABLATIONS",
    "COMPANION_V02_FIGURE_08_CORRECTED_RESEARCH_STORY",
)


def expected_pdf_paths(paper_root: str | Path) -> list[Path]:
    paper = Path(paper_root).resolve()
    return [
        *(paper / "figures" / f"{figure_id}.pdf" for figure_id in FIGURE_IDS),
        paper / "build/manuscript.pdf",
        paper / "build/supplement.pdf",
        paper / "ONE_PAGE_LAY_SUMMARY.pdf",
        paper / "graphical_abstract.pdf",
    ]


def _run(command: list[str], *, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n{completed.stderr}"
        )
    return completed.stdout


def _pdfinfo(path: Path, executable: str) -> dict[str, str]:
    raw = _run([executable, str(path)])
    return {
        key.strip(): value.strip()
        for line in raw.splitlines()
        if ":" in line
        for key, value in [line.split(":", 1)]
    }


def _render(path: Path, output_dir: Path, executable: str) -> list[Path]:
    destination = output_dir / path.stem
    destination.mkdir(parents=True, exist_ok=True)
    prefix = destination / "page"
    _run([executable, "-png", "-r", "120", str(path), str(prefix)])
    pages = sorted(destination.glob("page-*.png"))
    if not pages:
        raise RuntimeError(f"Poppler rendered no pages for {path}")
    return pages


def _page_metrics(path: Path) -> dict[str, Any]:
    with Image.open(path) as raw:
        image = raw.convert("RGB")
        grayscale = image.convert("L")
        difference = ImageChops.difference(image, Image.new("RGB", image.size, "white"))
        occupied = difference.convert("L").point(lambda value: 255 if value > 8 else 0)
        bbox = occupied.getbbox()
        stat = ImageStat.Stat(grayscale)
        return {
            "path": str(path),
            "pixels": [image.width, image.height],
            "grayscale_mean": stat.mean[0],
            "grayscale_stddev": stat.stddev[0],
            "nonwhite_bbox": None if bbox is None else list(bbox),
            "technically_blank": bbox is None or stat.stddev[0] < 0.5,
        }


def _contact_sheet(page_paths: list[Path], target: Path, *, label: str) -> None:
    thumbnails: list[Image.Image] = []
    try:
        for index, path in enumerate(page_paths, start=1):
            with Image.open(path) as raw:
                thumb = raw.convert("RGB")
                thumb.thumbnail((520, 700), Image.Resampling.LANCZOS)
            card = Image.new("RGB", (560, 760), "#1b2028")
            x = (card.width - thumb.width) // 2
            card.paste(thumb, (x, 36))
            draw = ImageDraw.Draw(card)
            draw.text((18, 12), f"{label} · page {index}", fill="white")
            thumbnails.append(card)
        columns = min(4, max(1, len(thumbnails)))
        rows = (len(thumbnails) + columns - 1) // columns
        sheet = Image.new("RGB", (columns * 560, rows * 760), "#0d1117")
        for index, card in enumerate(thumbnails):
            sheet.paste(card, ((index % columns) * 560, (index // columns) * 760))
        target.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(target, format="PNG", optimize=True)
    finally:
        for image in thumbnails:
            image.close()


def render_pdf_qa(
    *,
    paper_root: str | Path,
    output_path: str | Path,
    pdftoppm: str | None = None,
    pdfinfo: str | None = None,
) -> dict[str, Any]:
    paper = Path(paper_root).resolve()
    output = Path(output_path).resolve()
    renderer = pdftoppm or shutil.which("pdftoppm")
    inspector = pdfinfo or shutil.which("pdfinfo")
    if not renderer or not inspector:
        raise RuntimeError("Poppler pdftoppm/pdfinfo executables are required")
    expected = expected_pdf_paths(paper)
    if len(expected) != 7:
        raise RuntimeError("internal corrected PDF inventory is not exactly seven")
    records: list[dict[str, Any]] = []
    for path in expected:
        verify_sidecar(path)
        info = _pdfinfo(path, inspector)
        pages = _render(path, output, renderer)
        declared_pages = int(info.get("Pages", "0"))
        if declared_pages != len(pages):
            raise RuntimeError(f"pdfinfo/render page-count mismatch for {path}")
        metrics = [_page_metrics(page) for page in pages]
        contact = output / "contact_sheets" / f"{path.stem}.png"
        _contact_sheet(pages, contact, label=path.stem)
        atomic_write_text(
            contact.with_suffix(contact.suffix + ".sha256"),
            f"{sha256_file(contact)}  {contact.name}\n",
        )
        records.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "pages": declared_pages,
                "page_metrics": metrics,
                "contact_sheet": str(contact),
                "contact_sheet_sha256": sha256_file(contact),
            }
        )
    result = {
        "schema": "zws-reversible-null-phase-companion-v02-pdf-technical-qa-v1",
        "pdf_count": len(records),
        "rendered_page_count": sum(record["pages"] for record in records),
        "blank_page_count": sum(
            page["technically_blank"] for record in records for page in record["page_metrics"]
        ),
        "all_sidecars_and_page_counts_passed": True,
        "visual_review_required": True,
        "records": records,
    }
    atomic_write_json(output / "PDF_TECHNICAL_QA.json", result)
    return result
