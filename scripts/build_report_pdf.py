"""Render reports/report.md to a typeset PDF.

    python scripts/build_report_pdf.py

Markdown -> HTML via pandoc, HTML -> PDF via WeasyPrint, falling back to
headless Chrome when WeasyPrint is not installed. Kept as a script
rather than a Makefile one-liner because the stylesheet needs to travel with
it: the report is figure-heavy and the default rendering breaks images across
page boundaries.
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


# Headless Chrome is the fallback renderer. WeasyPrint gives a better result --
# it implements CSS Paged Media, so the @page rules above produce real page
# numbers -- but on macOS it needs Homebrew's pango, which is a lot of setup for
# one PDF. Chrome ships on nearly every machine, renders the same HTML and CSS,
# and only loses the page numbers.
CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    "google-chrome", "google-chrome-stable", "chromium", "chromium-browser",
    "microsoft-edge",
]


def _find_chrome() -> str | None:
    for cand in CHROME_CANDIDATES:
        if "/" in cand:
            if Path(cand).exists():
                return cand
        elif shutil.which(cand):
            return shutil.which(cand)
    return None


def _render_weasyprint(html: Path) -> bool:
    try:
        from weasyprint import HTML as WHTML
    except ImportError:
        return False
    # The stylesheet is embedded in the document, so no `stylesheets=` here.
    WHTML(filename=str(html), base_url=str(REPORTS)).write_pdf(str(PDF))
    return True


def _render_chrome(html: Path) -> bool:
    exe = _find_chrome()
    if exe is None:
        return False
    subprocess.run(
        [exe, "--headless=new", "--disable-gpu", "--no-sandbox",
         "--no-pdf-header-footer", f"--print-to-pdf={PDF}",
         html.resolve().as_uri()],
        check=True, capture_output=True,
    )
    return PDF.exists()


def build() -> None:
    if not MD.exists():
        raise SystemExit(f"{MD} not found")

    # No --standalone: pandoc's template injects its own <h1 class="title">,
    # which would duplicate the H1 already at the top of the markdown.
    #
    # `+footnotes` matters -- §1.1 and §11.3 cite their sources as footnotes,
    # and the bare gfm reader renders them as literal `[^1]` text.
    # Older pandoc (< 2.11) rejects `gfm+footnotes` outright, so try the
    # readers in descending order of fidelity and take the first that runs.
    body = None
    for reader in ("gfm+footnotes", "commonmark_x", "gfm"):
        proc = subprocess.run(["pandoc", str(MD), "-f", reader, "-t", "html5"],
                              capture_output=True, text=True)
        if proc.returncode == 0:
            body = proc.stdout
            if reader == "gfm":
                print("note: this pandoc cannot render footnotes; the source "
                      "citations in §1.1 and §11.3 will appear as [^n] markers. "
                      "Upgrade pandoc to fix.", file=sys.stderr)
            break
    if body is None:
        raise SystemExit("pandoc failed on every reader; is it installed?")

    # Written next to the figures rather than in /tmp: Chrome resolves relative
    # image paths against the document's own location and has no equivalent of
    # WeasyPrint's base_url.
    html = REPORTS / "_report_build.html"
    html.write_text(
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        "<title>FlightRisk NYC</title><style>" + CSS + "</style></head><body>"
        + body + "</body></html>")

    try:
        if _render_weasyprint(html):
            engine = "WeasyPrint"
        elif _render_chrome(html):
            engine = "headless Chrome (no page numbers -- "
            engine += "`pip install weasyprint` for those)"
        else:
            raise SystemExit(
                "No PDF renderer found. Either:\n"
                "  pip install weasyprint      (macOS also needs: brew install pango)\n"
                "or install Google Chrome, which this script will use "
                "automatically.\n"
                f"The typeset HTML has been left at {html} in the meantime.")
    finally:
        if PDF.exists():
            html.unlink(missing_ok=True)

    print(f"wrote {PDF} ({PDF.stat().st_size / 1e6:.1f} MB) via {engine}")


if __name__ == "__main__":
    sys.exit(build())
