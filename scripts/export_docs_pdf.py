from __future__ import annotations

import html
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "GNN第二版从零详解"
OUTPUT = DOCS / "PDF导出"


def inline_markup(value: str) -> str:
    value = html.escape(value, quote=False)
    value = re.sub(r"`([^`]+)`", r"<code>\1</code>", value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", value)
    value = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", value)
    value = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", value)
    return value


def markdown_to_html(source: str, title: str) -> str:
    lines = source.splitlines()
    blocks: list[str] = []
    paragraph: list[str] = []
    list_tag: str | None = None
    in_code = False
    code_lines: list[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            blocks.append(f"<p>{'<br>'.join(inline_markup(line) for line in paragraph)}</p>")
            paragraph = []

    def close_list() -> None:
        nonlocal list_tag
        if list_tag:
            blocks.append(f"</{list_tag}>")
            list_tag = None

    index = 0
    while index < len(lines):
        line = lines[index]
        if in_code:
            if line.strip().startswith(("```", "~~~")):
                blocks.append(f"<pre><code>{html.escape(chr(10).join(code_lines))}</code></pre>")
                code_lines = []
                in_code = False
            else:
                code_lines.append(line)
            index += 1
            continue

        if line.strip().startswith(("```", "~~~")):
            flush_paragraph()
            close_list()
            in_code = True
            index += 1
            continue

        if not line.strip():
            flush_paragraph()
            close_list()
            index += 1
            continue

        heading = re.match(r"^(#{1,6})\s+(.+?)\s*#*$", line)
        if heading:
            flush_paragraph()
            close_list()
            level = len(heading.group(1))
            blocks.append(f"<h{level}>{inline_markup(heading.group(2))}</h{level}>")
            index += 1
            continue

        if line.lstrip().startswith(">"):
            flush_paragraph()
            close_list()
            quote = line.lstrip()[1:].strip()
            blocks.append(f"<blockquote>{inline_markup(quote)}</blockquote>")
            index += 1
            continue

        if line.startswith("|") and index + 1 < len(lines) and lines[index + 1].startswith("|"):
            flush_paragraph()
            close_list()
            rows: list[list[str]] = []
            while index < len(lines) and lines[index].startswith("|"):
                cells = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", cell) for cell in cells):
                    rows.append(cells)
                index += 1
            if rows:
                head = "".join(f"<th>{inline_markup(cell)}</th>" for cell in rows[0])
                body = "".join(
                    "<tr>" + "".join(f"<td>{inline_markup(cell)}</td>" for cell in row) + "</tr>"
                    for row in rows[1:]
                )
                blocks.append(f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>")
            continue

        unordered = re.match(r"^\s*[-*+]\s+(.+)$", line)
        ordered = re.match(r"^\s*\d+[.)]\s+(.+)$", line)
        if unordered or ordered:
            tag = "ul" if unordered else "ol"
            content = (unordered or ordered).group(1)
            flush_paragraph()
            if list_tag != tag:
                close_list()
                list_tag = tag
                blocks.append(f"<{tag}>")
            blocks.append(f"<li>{inline_markup(content)}</li>")
            index += 1
            continue

        if list_tag:
            close_list()
        paragraph.append(line)
        index += 1

    if in_code:
        blocks.append(f"<pre><code>{html.escape(chr(10).join(code_lines))}</code></pre>")
    flush_paragraph()
    close_list()
    body = "\n".join(blocks)
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
@page {{ size: A4; margin: 1.8cm 1.7cm 1.8cm 1.7cm; }}
body {{ font-family: "Noto Sans CJK SC", "Microsoft YaHei", sans-serif; font-size: 10.5pt; line-height: 1.55; color: #111; }}
h1 {{ font-size: 21pt; border-bottom: 1px solid #999; padding-bottom: 5pt; margin-top: 0; }}
h2 {{ font-size: 16pt; margin-top: 18pt; border-bottom: 0.5px solid #ccc; }}
h3 {{ font-size: 12.5pt; margin-top: 14pt; }}
p {{ margin: 6pt 0; }}
blockquote {{ border-left: 3px solid #999; margin: 8pt 0; padding: 3pt 10pt; color: #333; }}
code, pre {{ font-family: "Noto Sans Mono CJK SC", "DejaVu Sans Mono", monospace; }}
code {{ background: #f0f0f0; padding: 0 2pt; }}
pre {{ background: #f4f4f4; border: 0.5px solid #ccc; padding: 7pt; white-space: pre-wrap; font-size: 8.5pt; }}
table {{ border-collapse: collapse; width: 100%; margin: 8pt 0; font-size: 8.5pt; }}
th, td {{ border: 0.5px solid #777; padding: 3pt 4pt; vertical-align: top; }}
th {{ background: #e8e8e8; }}
li {{ margin: 2pt 0; }}
</style></head><body>{body}</body></html>"""


def export_one(markdown_path: Path, temporary_dir: Path) -> None:
    html_path = temporary_dir / f"{markdown_path.stem}.html"
    html_path.write_text(markdown_to_html(markdown_path.read_text(encoding="utf-8"), markdown_path.stem), encoding="utf-8")
    profile = temporary_dir / f"profile_{markdown_path.stem}"
    pdf_dir = temporary_dir / f"pdf_{markdown_path.stem}"
    pdf_dir.mkdir()
    result = subprocess.run(
        [
            "libreoffice",
            "--headless",
            f"-env:UserInstallation=file://{profile}",
            "--convert-to",
            "pdf",
            "--outdir",
            str(pdf_dir),
            str(html_path),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"LibreOffice export failed for {markdown_path}: {result.stdout}")
    generated = pdf_dir / f"{markdown_path.stem}.pdf"
    if not generated.exists():
        raise RuntimeError(f"LibreOffice did not create {generated}")
    shutil.copy2(generated, OUTPUT / generated.name)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gnn_v2_pdf_") as temporary:
        temporary_dir = Path(temporary)
        for markdown_path in sorted(DOCS.glob("*.md")):
            export_one(markdown_path, temporary_dir)
            print(f"exported {markdown_path.name}")


if __name__ == "__main__":
    main()
