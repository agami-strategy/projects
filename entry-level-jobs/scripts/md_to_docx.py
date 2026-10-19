"""Convert a simple Markdown document (headings, paragraphs, pipe tables, *italic*) to .docx.

Covers exactly the constructs used in docs/methodology.md; not a general Markdown converter.

Usage:
    python scripts/md_to_docx.py docs/methodology.md docs/methodology.docx
"""
# %% Imports
# Utility for sharing the methodology write-up as a Word file. It is a small hand-written parser for
# the subset of Markdown used in docs/: headings (#, ##, ...), plain paragraphs, pipe tables, and the
# inline marks **bold**, *italic* and `code`. Lists, links, images, block quotes and fenced code are
# not handled and would come through as plain paragraph text.
from __future__ import annotations

import re
import sys
from pathlib import Path

# python-docx: Document builds the file; the enum modules give alignment constants; OxmlElement / qn
# are the low-level XML hooks needed for cell shading, which python-docx has no high-level API for.
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

# %% Inline markup
# One regex with a capturing group so re.split returns the delimiters as well as the text between
# them. Alternatives are tried left to right, so **bold** must precede *italic* or a bold span would
# be read as an italic span containing asterisks. Nesting (bold inside italic) is not supported.
INLINE = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)")


# add_inline: append runs to a python-docx paragraph, one per chunk, with the matching formatting.
# re.split yields empty strings between adjacent delimiters; those are skipped. Code spans are set
# in Consolas by changing the run font rather than using a character style.
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


# %% Table cell shading via raw OOXML
# python-docx exposes no property for cell background, so the WordprocessingML element
# <w:shd w:val="clear" w:color="auto" w:fill="RRGGBB"/> is appended to the cell's properties (tcPr)
# by hand. qn() expands the "w:" prefix to the full namespace. hex_fill is a 6-digit hex colour.
def set_cell_shading(cell, hex_fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tc_pr.append(shd)


# %% Pipe table -> Word table
# `rows` is a list of cell-text lists, already split on "|". Row 0 is the header. Row 1 is treated as
# the Markdown alignment separator (e.g. "---" or ":--:") when its first cell contains only "-", ":"
# and spaces, in which case the body starts at row 2; otherwise the table has no separator line and
# the body starts at row 1. (The conditional expression binds to the second tuple element only.)
# Header cells are bold on a light grey (E7E6E6) fill. Body cells that look numeric ("1,234", "12.5%",
# "about 1,200", or a lone "-" placeholder) are right-aligned, except in the first column, which is
# the row label. Short rows are padded with "" so ragged tables do not raise IndexError.
# The empty paragraph after the table stops Word from gluing consecutive tables together.
def add_table(doc, rows: list[list[str]]) -> None:
    header, body = rows[0], rows[2:] if rows[1] and set(rows[1][0]) <= set("-: ") else rows[1:]
    table = doc.add_table(rows=1 + len(body), cols=len(header))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, h in enumerate(header):
        cell = table.rows[0].cells[j]
        # cell.text = "" clears the default empty run before add_inline builds formatted runs
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


# %% Document conversion: a line-by-line state machine
# Base formatting: Normal style Calibri 11 pt; Heading 1-3 recoloured to a dark blue (1F3A5F) instead
# of Word's default theme blue. The parser then walks the Markdown lines keeping two buffers:
#   para_buf   consecutive non-blank, non-table, non-heading lines; joined with spaces on flush, so
#              a hard-wrapped Markdown paragraph becomes one Word paragraph
#   table_buf  consecutive lines starting with "|"; parsed and rendered on flush
# A blank line or a heading flushes both; a table line flushes the paragraph; a text line flushes the
# table. Heading depth: "# " becomes the document Title style (level 0), "## " Heading 1, and so on,
# so the file's single top-level heading acts as the title.
def convert(md_path: Path, out_path: Path) -> None:
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)
    for s in ("Heading 1", "Heading 2", "Heading 3"):
        doc.styles[s].font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)

    lines = md_path.read_text(encoding="utf-8").splitlines()
    i, para_buf, table_buf = 0, [], []

    # closures over doc / para_buf / table_buf; .clear() mutates the shared list in place
    def flush_para():
        if para_buf:
            p = doc.add_paragraph()
            add_inline(p, " ".join(s.strip() for s in para_buf))
            para_buf.clear()

    # strip the outer "|" then split on "|"; leading/trailing whitespace in each cell is removed
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
            # level = number of leading "#" characters
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
    # flush whatever is still buffered at end of file
    flush_para(); flush_table()
    doc.save(out_path)


# %% Command-line entry point: input .md path, output .docx path
if __name__ == "__main__":
    convert(Path(sys.argv[1]), Path(sys.argv[2]))
    print("wrote", sys.argv[2])
