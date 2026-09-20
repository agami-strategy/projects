"""Convert a simple Markdown document (headings, paragraphs, pipe tables, *italic*) to .docx.

Covers exactly the constructs used in docs/methodology.md; not a general Markdown converter.

Usage:
    python scripts/md_to_docx.py docs/methodology.md docs/methodology.docx
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

INLINE = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)")


def add_inline(par, text: str) -> None:
    for chunk in INLINE.split(text):
        if not chunk:
            continue
        if chunk.startswith("**"):
            par.add_run(chunk[2:-2]).bold = True
        elif chunk.startswith("*"):
            par.add_run(chunk[1:-1]).italic = True
        elif chunk.startswith("`"):
            r = par.add_run(chunk[1:-1])
            r.font.name = "Consolas"
        else:
            par.add_run(chunk)


def set_cell_shading(cell, hex_fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tc_pr.append(shd)


def add_table(doc, rows: list[list[str]]) -> None:
    header, body = rows[0], rows[2:] if rows[1] and set(rows[1][0]) <= set("-: ") else rows[1:]
    table = doc.add_table(rows=1 + len(body), cols=len(header))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, h in enumerate(header):
        cell = table.rows[0].cells[j]
        cell.text = ""
        add_inline(cell.paragraphs[0], h)
        for r in cell.paragraphs[0].runs:
            r.bold = True
        set_cell_shading(cell, "E7E6E6")
    for i, row in enumerate(body, start=1):
        for j in range(len(header)):
            cell = table.rows[i].cells[j]
            cell.text = ""
            val = row[j] if j < len(row) else ""
            add_inline(cell.paragraphs[0], val)
            if j > 0 and re.fullmatch(r"[\d,.\s%]+|about [\d,.]+|-", val or ""):
                cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
    doc.add_paragraph()


def convert(md_path: Path, out_path: Path) -> None:
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)
    for s in ("Heading 1", "Heading 2", "Heading 3"):
        doc.styles[s].font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)

    lines = md_path.read_text(encoding="utf-8").splitlines()
    i, para_buf, table_buf = 0, [], []

    def flush_para():
        if para_buf:
            p = doc.add_paragraph()
            add_inline(p, " ".join(s.strip() for s in para_buf))
            para_buf.clear()

    def flush_table():
        if table_buf:
            rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in table_buf]
            add_table(doc, rows)
            table_buf.clear()

    while i < len(lines):
        ln = lines[i]
        if ln.startswith("|"):
            flush_para()
            table_buf.append(ln)
        elif ln.startswith("#"):
            flush_para(); flush_table()
            level = len(ln) - len(ln.lstrip("#"))
            text = ln[level:].strip()
            if level == 1:
                doc.add_heading(text, 0)
            else:
                doc.add_heading(text, level - 1)
        elif not ln.strip():
            flush_para(); flush_table()
        else:
            flush_table()
            para_buf.append(ln)
        i += 1
    flush_para(); flush_table()
    doc.save(out_path)


if __name__ == "__main__":
    convert(Path(sys.argv[1]), Path(sys.argv[2]))
    print("wrote", sys.argv[2])
