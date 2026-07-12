"""report.md -> docs/report.html (KaTeX via CDN, auto-render) -> docs/report.pdf (headless Chrome)."""

import markdown
import pathlib
import subprocess

md = pathlib.Path("docs/report.md").read_text()
body = markdown.markdown(md, extensions=["tables", "fenced_code"])

html = f"""<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
 onload="renderMathInElement(document.body,{{delimiters:[{{left:'$$',right:'$$',display:true}},{{left:'$',right:'$',display:false}}]}});"></script>
<style>
body {{ font-family: -apple-system, 'Helvetica Neue', sans-serif; font-size: 11.5px;
       line-height: 1.45; max-width: 820px; margin: 24px auto; color: #1a1a1a; }}
h1 {{ font-size: 20px; }} h2 {{ font-size: 16px; margin-top: 22px; }}
h3 {{ font-size: 13px; margin-top: 16px; }}
table {{ border-collapse: collapse; margin: 10px 0; }}
th, td {{ border: 1px solid #bbb; padding: 3px 8px; font-size: 10.5px; }}
th {{ background: #f0f0f0; }}
img {{ max-width: 100%; margin: 8px 0; }}
code {{ background: #f4f4f4; padding: 1px 4px; border-radius: 3px; font-size: 10px; }}
</style></head><body>{body}</body></html>"""
pathlib.Path("docs/report.html").write_text(html)

subprocess.run(
    [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--virtual-time-budget=15000",
        "--print-to-pdf=docs/report.pdf",
        "docs/report.html",
    ],
    check=True,
)
print("wrote docs/report.pdf")
