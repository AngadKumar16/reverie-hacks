"""Render reports/report.md to a typeset PDF.

    python scripts/build_report_pdf.py

Two stages, each with a fallback, because the original toolchain needed two
system packages and the report is a *required submission deliverable* -- a PDF
that can only be rebuilt on the one machine that has pandoc and a working
libpango is a liability, not a build step.

    markdown -> HTML   pandoc, else the pure-Python `markdown` package
    HTML     -> PDF    WeasyPrint, else headless Chrome

The stylesheet travels with the script rather than living in the Makefile: the
report is figure-heavy and the default rendering breaks images across page
boundaries. Both backends were checked against the same stylesheet, including
the CSS Paged Media rules -- page numbering via `@bottom-center` and the
suppressed number on `@page :first` come out the same either way.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
MD = REPORTS / "report.md"
HTML = Path(tempfile.gettempdir()) / "_flightrisk_report.html"
PDF = REPORTS / "report.pdf"

CSS = """
@page {
  size: A4;
  margin: 20mm 17mm 18mm 17mm;
  @bottom-center {
    content: counter(page);
    font-family: "DejaVu Sans", sans-serif;
    font-size: 8.5pt;
    color: #8a8a8a;
  }
}
@page :first { @bottom-center { content: ""; } }

body {
  font-family: "DejaVu Serif", Georgia, serif;
  font-size: 10pt;
  line-height: 1.52;
  color: #1b1b1b;
  hyphens: auto;
  text-align: justify;
}

h1 {
  font-family: "DejaVu Sans", sans-serif;
  font-size: 26pt;
  line-height: 1.15;
  margin: 0 0 2mm 0;
  color: #21323d;
  text-align: left;
}
h1 + h3 {
  font-family: "DejaVu Sans", sans-serif;
  font-size: 13pt;
  font-weight: 400;
  color: #5a6b76;
  margin: 0 0 6mm 0;
  border: none;
  text-align: left;
}
h2 {
  font-family: "DejaVu Sans", sans-serif;
  font-size: 14.5pt;
  color: #21323d;
  margin: 9mm 0 3mm 0;
  padding-bottom: 1.6mm;
  border-bottom: 1.6pt solid #3b6978;
  break-after: avoid;
  text-align: left;
}
h3 {
  font-family: "DejaVu Sans", sans-serif;
  font-size: 11.5pt;
  color: #3b6978;
  margin: 6mm 0 2mm 0;
  break-after: avoid;
  text-align: left;
}

p { margin: 0 0 2.6mm 0; orphans: 2; widows: 2; }
ul, ol { margin: 0 0 3mm 0; padding-left: 6mm; }
li { margin-bottom: 1.4mm; }

strong { color: #0d0d0d; }
em { color: #333; }

code {
  font-family: "DejaVu Sans Mono", monospace;
  font-size: 8.6pt;
  background: #f2f4f5;
  padding: 0.4mm 1mm;
  border-radius: 2px;
}
pre {
  background: #f6f8f9;
  border-left: 2.5pt solid #3b6978;
  padding: 2.6mm 3.5mm;
  font-size: 8.4pt;
  line-height: 1.42;
  overflow-wrap: break-word;
  white-space: pre-wrap;
  break-inside: avoid;
  margin: 0 0 3.5mm 0;
}
pre code { background: none; padding: 0; }

table {
  border-collapse: collapse;
  width: 100%;
  margin: 2mm 0 4.5mm 0;
  font-family: "DejaVu Sans", sans-serif;
  font-size: 8.6pt;
  break-inside: avoid;
}
th {
  background: #21323d;
  color: #fff;
  text-align: left;
  padding: 1.8mm 2.2mm;
  font-weight: 600;
}
td { padding: 1.5mm 2.2mm; border-bottom: 0.4pt solid #d8dee1; }
tbody tr:nth-child(even) { background: #f5f7f8; }

img {
  max-width: 100%;
  display: block;
  margin: 3mm auto 4mm auto;
  break-inside: avoid;
}

blockquote {
  border-left: 2.5pt solid #c44e52;
  margin: 0 0 3mm 0;
  padding-left: 4mm;
  color: #444;
}

hr { border: none; border-top: 0.5pt solid #ccd4d8; margin: 6mm 0; }
a { color: #2b5f7a; text-decoration: none; word-break: break-all; }
"""


CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "google-chrome", "chromium", "chromium-browser", "microsoft-edge",
]


def _find_chrome() -> str | None:
    for c in CHROME_CANDIDATES:
        if Path(c).exists():
            return c
        found = shutil.which(c)
        if found:
            return found
    return None


def md_to_html(md_text: str) -> str:
    """Markdown body -> HTML. pandoc if present, else python-markdown."""
    if shutil.which("pandoc"):
        # No --standalone: pandoc's template injects its own
        # <h1 class="title">, duplicating the H1 already in the markdown.
        return subprocess.run(
            ["pandoc", str(MD), "-f", "gfm", "-t", "html5"],
            check=True, capture_output=True, text=True,
        ).stdout
    try:
        import markdown
    except ImportError:
        raise SystemExit(
            "Need either pandoc on PATH or `pip install markdown` to turn the "
            "report into HTML.")
    return markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "footnotes", "attr_list",
                    "sane_lists", "md_in_html"],
        extension_configs={"footnotes": {"BACKLINK_TEXT": "&#8617;"}},
    )


def html_to_pdf(html_path: Path, css_path: Path) -> str:
    """HTML -> PDF. WeasyPrint if it imports, else headless Chrome."""
    try:
        from weasyprint import CSS as WCSS, HTML as WHTML
    except Exception as exc:                      # missing pango/cairo, usually
        chrome = _find_chrome()
        if chrome is None:
            raise SystemExit(
                f"WeasyPrint is unavailable ({exc.__class__.__name__}) and no "
                "Chrome/Chromium was found to fall back to. Install one of "
                "them, or `brew install pango` to repair WeasyPrint.")
        # Chrome will not read a separate stylesheet argument, so inline it.
        doc = html_path.read_text().replace(
            "</head>", f"<style>{css_path.read_text()}</style></head>")
        html_path.write_text(doc)
        subprocess.run(
            [chrome, "--headless", "--disable-gpu", "--no-sandbox",
             "--no-pdf-header-footer", "--print-to-pdf-no-header",
             f"--print-to-pdf={PDF}", html_path.as_uri()],
            check=True, capture_output=True,
        )
        return f"headless Chrome ({Path(chrome).name})"

    WHTML(filename=str(html_path), base_url=str(REPORTS)).write_pdf(
        str(PDF), stylesheets=[WCSS(filename=str(css_path))])
    return "WeasyPrint"


def build() -> None:
    if not MD.exists():
        raise SystemExit(f"{MD} not found")

    body = md_to_html(MD.read_text())
    HTML.write_text(
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        '<base href="' + REPORTS.as_uri() + '/">'
        "<title>FlightRisk NYC</title></head><body>" + body + "</body></html>"
    )

    css_file = Path(tempfile.gettempdir()) / "_flightrisk_report.css"
    css_file.write_text(CSS)

    backend = html_to_pdf(HTML, css_file)

    HTML.unlink(missing_ok=True)
    css_file.unlink(missing_ok=True)
    print(f"wrote {PDF} ({PDF.stat().st_size / 1e6:.1f} MB) via {backend}")


if __name__ == "__main__":
    sys.exit(build())
