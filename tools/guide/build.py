"""Build the field guide in docs/ from pages, notes and measured results.

    python -m tools.guide.build           write everything
    python -m tools.guide.build --check   fail if docs/ is out of date

No third-party packages. Pages are HTML fragments in tools/guide/pages. Numbers
in the prose are written as {{sci headline.noise_floor}} and filled in from
results/*.json, so the text cannot drift away from the measurements.
"""

import html
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DOCS = ROOT / "docs"
BRAND = "Ten Dots"

CHAPTERS = [
    "question", "spring", "network", "derivatives", "loss", "training",
    "results", "breaks", "inverse", "byhand", "ledger",
]
EXTRAS = ["route", "log"]
NAV = [("index", "Contents"), ("route", "Route"), ("source", "Source"), ("questions", "Questions"), ("log", "Build log")]

FONTS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
    '<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&amp;'
    "family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,500;0,8..60,600;0,8..60,700;"
    '1,8..60,400;1,8..60,500&amp;display=swap" rel="stylesheet">'
)


# ---- results -------------------------------------------------------------


def load_results():
    out = {}
    for path in sorted((ROOT / "results").glob("*.json")):
        out[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    return out


def numbers(results):
    """Flat lookup for {{...}}. Ablations become name.setting.key medians."""
    table = {}

    def walk(prefix, value):
        if isinstance(value, dict):
            for key, inner in value.items():
                walk(f"{prefix}.{key}", inner)
        elif isinstance(value, (int, float)):
            table[prefix] = value

    for name, payload in results.items():
        if name == "headline":
            walk("headline", {k: v for k, v in payload.items() if k not in ("rows", "t", "truth")})
            ratios = payload["ratio_per_seed"]
            table["headline.ratio_min"], table["headline.ratio_max"] = min(ratios), max(ratios)
            table["headline.ratio_median"] = statistics.median(ratios)
            m = payload["median_mse"]
            table["headline.vs_zero"] = m["zero"] / m["pinn"]
            table["headline.ratio_lbfgs"] = m["plain_lbfgs"] / m["pinn"]
            table["headline.vs_floor"] = m["pinn"] / m["classical"]
            for method in m:
                values = [r["mse"] for r in payload["rows"] if r["method"] == method]
                table[f"headline.min.{method}"], table[f"headline.max.{method}"] = min(values), max(values)
                seconds = [r["seconds"] for r in payload["rows"] if r["method"] == method and "seconds" in r]
                if seconds:
                    table[f"headline.seconds.{method}"] = statistics.median(seconds)
                data = [r["data_loss"] for r in payload["rows"] if r["method"] == method and "data_loss" in r]
                if data:
                    table[f"headline.data_loss.{method}"] = statistics.median(data)
        elif name == "inverse":
            for noise in {r["noise"] for r in payload}:
                for key in ("c", "k", "mse"):
                    values = [r[key] for r in payload if r["noise"] == noise]
                    table[f"inverse.{noise}.{key}.min"] = min(values)
                    table[f"inverse.{noise}.{key}.max"] = max(values)
                    table[f"inverse.{noise}.{key}"] = statistics.median(values)
        else:
            rows = payload["rows"] if isinstance(payload, dict) else payload
            short = name.replace("ablation_", "")
            for setting in {r["setting"] for r in rows}:
                for key in ("mse", "classical", "data_loss", "physics_loss", "dense_residual", "seconds"):
                    values = [r[key] for r in rows if r["setting"] == setting]
                    table[f"{short}.{setting}.{key}"] = statistics.median(values)
                    table[f"{short}.{setting}.{key}.max"] = max(values)
                    table[f"{short}.{setting}.{key}.min"] = min(values)
    return table


def fmt(kind, value):
    if kind == "sci":
        if 0.01 <= abs(value) < 1000:
            return f"{value:.2g}" if abs(value) < 100 else f"{value:.0f}"
        mantissa, exponent = f"{value:.1e}".split("e")
        return f"{mantissa}&nbsp;&times;&nbsp;10<sup>{int(exponent)}</sup>".replace("<sup>-", "<sup>&minus;")
    if kind == "int":  # two significant figures, with thousands separators
        digits = max(0, len(str(int(abs(value)))) - 2)
        return f"{round(value, -digits):,.0f}"
    if kind.startswith("fix"):
        return f"{value:.{int(kind[3:])}f}"
    raise ValueError(f"unknown number format {kind}")


def fill(text, table):
    def swap(match):
        kind, key = match.group(1), match.group(2)
        if key not in table:
            raise KeyError(f"no result called {key}")
        return fmt(kind, table[key])

    return re.sub(r"\{\{(\w+) ([^}]+)\}\}", swap, text)


# ---- pages ---------------------------------------------------------------


def read_page(name):
    raw = (HERE / "pages" / f"{name}.html").read_text(encoding="utf-8")
    head, body = raw.split("\n---\n", 1)
    meta = dict(line.split(": ", 1) for line in head.strip().splitlines())
    return meta, body


def shell(name, title, body, scripts=(), repo=""):
    nav = "".join(
        f'<a href="{n}.html"{" aria-current=page" if n == name else ""}>{label}</a>' for n, label in NAV
    )
    tags = "\n".join(f'<script src="assets/{s}"></script>' for s in ("site.js", *scripts))
    code = f'<a href="{repo}">Repository</a>' if repo else ""
    full = BRAND if name == "index" else f"{title} &middot; {BRAND}"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>{full}</title>
<meta name="description" content="A physics-informed neural network sees ten noisy dots and predicts what comes next. Built from first principles, with every step shown.">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3E%3Cpath d='M1 8c2-9 4-9 6 0s4 6 8 0' fill='none' stroke='%232b4cf2' stroke-width='2'/%3E%3C/svg%3E">
{FONTS}
<link rel="stylesheet" href="assets/style.css">
</head>
<body>
<a class="skip" href="#main">Skip to content</a>
<header class="topbar"><a class="brand" href="index.html">{BRAND} <span>/ field guide</span></a><nav aria-label="Site">{nav}</nav></header>
<main id="main" class="page">
{body}
</main>
<footer class="foot"><span>{BRAND} / field guide</span><span>{code}<a href="index.html">Contents</a><a href="source.html">Source</a><a href="questions.html">Questions</a></span></footer>
{tags}
</body>
</html>
"""


FIG_SCRIPTS = ("engine.js", "results-data.js", "figkit.js", "figures.js", "figures-results.js")


class FigureNumbers:
    """Number figures in reading order across the whole guide."""

    def __init__(self):
        self.n = 0

    def __call__(self, body):
        def swap(match):
            label = f"FIG_{self.n:03d}"
            self.n += 1
            return f'<span class="fig-label">{label}</span>\n<div class="fig-body"></div>'

        return re.sub(r"<!--fig-->", swap, body)


def build_pages(table, repo):
    out, number = {}, FigureNumbers()
    metas = {name: read_page(name)[0] for name in CHAPTERS}

    meta, body = read_page("index")
    toc = "\n".join(
        f'<li><a href="{n}.html"><span class="n">{i:02d}</span><span class="t">{metas[n]["title"]}</span>'
        f'<span class="h">{metas[n]["hint"]}</span></a></li>'
        for i, n in enumerate(CHAPTERS, 1)
    )
    body = number(fill(body, table)).replace("<!--toc-->", f'<ol class="toc">\n{toc}\n</ol>')
    out["index.html"] = shell("index", meta["title"], body, FIG_SCRIPTS, repo)

    for i, name in enumerate(CHAPTERS, 1):
        meta, body = read_page(name)
        body = number(fill(body, table))
        before = (f"{CHAPTERS[i - 2]}.html", f"{i - 1:02d} &middot; {metas[CHAPTERS[i - 2]]['short']}") if i > 1 else ("index.html", "Contents")
        after = (f"{CHAPTERS[i]}.html", f"{i + 1:02d} &middot; {metas[CHAPTERS[i]]['short']}") if i < len(CHAPTERS) else ("index.html", "Contents")
        page = f"""<div class="divider">{i:02d} &middot; {meta['short']}</div>
<header class="chapter-head"><span class="num">Chapter {i:02d}</span><h1>{meta['title']}</h1><p class="lede">{fill(meta['lede'], table)}</p></header>
{body}
<nav class="pager measure" aria-label="Chapters"><a href="{before[0]}">&larr; {before[1]}</a><a href="{after[0]}">{after[1]} &rarr;</a></nav>"""
        out[f"{name}.html"] = shell(name, meta["title"], page, FIG_SCRIPTS, repo)

    for name in EXTRAS:
        meta, body = read_page(name)
        body = number(fill(body, table))
        if name == "log":
            body = body.replace("<!--log-->", build_log(repo))
        page = f"""<div class="divider">{meta['short']}</div>
<header class="chapter-head"><span class="num">{meta['short']}</span><h1>{meta['title']}</h1><p class="lede">{fill(meta['lede'], table)}</p></header>
{body}"""
        out[f"{name}.html"] = shell(name, meta["title"], page, (), repo)

    for name, scripts in (("source", ("source-data.js", "source.js")), ("questions", ("questions-data.js", "questions.js"))):
        meta, body = read_page(name)
        out[f"{name}.html"] = shell(name, meta["title"], fill(body, table), scripts, repo)

    questions = (HERE / "questions-data.js").read_text(encoding="utf-8")
    out["assets/questions-data.js"] = fill(questions, table)
    return out


# ---- annotated source ----------------------------------------------------


def inline(text):
    text = html.escape(text, quote=False)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    return re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<i>\1</i>", text)


def build_source():
    files = []
    for notes in sorted((HERE / "annotations").glob("*.txt")):
        path = notes.stem.replace("__", "/")
        # __init__.py would be ambiguous, so no note file is written for it.
        lines = (ROOT / path).read_text(encoding="utf-8").rstrip("\n").split("\n")
        head, *blocks = re.split(r"^== ", notes.read_text(encoding="utf-8"), flags=re.M)
        meta = dict(re.findall(r"^@(\w+)\s+(.+)$", head, flags=re.M))
        starts, position = [], 0
        for block in blocks:
            anchor, note = (block.split("\n", 1) + [""])[:2]
            for i in range(position, len(lines)):
                if anchor.strip() in lines[i]:
                    starts.append((i, note))
                    position = i + 1
                    break
            else:
                raise SystemExit(f"{notes.name}: anchor not found in order: {anchor!r}")
        if not starts or starts[0][0] != 0:
            raise SystemExit(f"{notes.name}: the first anchor must match line 1")
        ends = [s for s, _ in starts[1:]] + [len(lines)]
        files.append(
            {
                "path": path,
                "title": meta["title"],
                "group": meta["group"],
                "order": int(meta.get("order", 99)),
                "blocks": [
                    {
                        "line": start + 1,
                        "code": lines[start:end],
                        "note": "".join(
                            f"<p>{inline(' '.join(p.split()))}</p>" for p in note.strip().split("\n\n") if p.strip()
                        ),
                    }
                    for (start, note), end in zip(starts, ends)
                ],
            }
        )
    files.sort(key=lambda f: f["order"])
    for f in files:
        del f["order"]
    return "window.SOURCE = " + json.dumps(files, ensure_ascii=False, indent=0) + ";\n"


def check_links(pages, source_js):
    """Every source.html#path@text link must point at a real line."""
    files = {f["path"]: "\n".join(c for b in f["blocks"] for c in b["code"]) for f in json.loads(source_js[16:-2])}
    for name, page in pages.items():
        for path, text in re.findall(r"source\.html#([^@\"']+)@([^\"']+)[\"']", page):
            if path not in files or html.unescape(text) not in files[path]:
                raise SystemExit(f"{name}: broken source link {path}@{text}")
        for target in re.findall(r'href="([a-z]+)\.html', page):
            if f"{target}.html" not in pages:
                raise SystemExit(f"{name}: link to missing page {target}.html")


# ---- build log -----------------------------------------------------------


def git(*args):
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    return result.stdout.strip()


def repo_url():
    url = git("remote", "get-url", "origin")
    match = re.match(r"(?:https://github\.com/|git@github\.com:)(.+?)(?:\.git)?$", url)
    return f"https://github.com/{match.group(1)}" if match else ""


def build_log(repo):
    """One entry per commit, with the explanation from log-notes.txt."""
    notes = {}
    for block in re.split(r"^== ", (HERE / "log-notes.txt").read_text(encoding="utf-8"), flags=re.M)[1:]:
        subject, text = block.split("\n", 1)
        notes[subject.strip()] = text.strip()
    rows, entries = git("log", "--reverse", "--format=%h%x09%ad%x09%s", "--date=format:%d %b %Y"), []
    for i, line in enumerate(rows.splitlines(), 1):
        sha, date, subject = line.split("\t", 2)
        files = git("show", "--stat=200", "--format=", sha).splitlines()
        stat = files[-1].strip() if files else ""
        link = f'<a href="{repo}/commit/{sha}">{sha}</a>' if repo else sha
        text = notes.get(subject, "")
        paragraphs = []
        for p in text.split("\n\n"):
            p = p.strip()
            if p.startswith("$ "):
                paragraphs.append("<pre>" + html.escape(p[2:]) + "</pre>")
            elif p:
                paragraphs.append(f"<p>{inline(' '.join(p.split()))}</p>")
        entries.append(
            f'<div><span class="lab step">Step {i:02d}</span><div><h4>{html.escape(subject)}</h4>'
            f'{"".join(paragraphs)}<div class="commit">{link} &middot; {date} &middot; {html.escape(stat)}</div></div></div>'
        )
    return '<div class="ledger">\n' + "\n".join(entries) + "\n</div>"


# ---- main ----------------------------------------------------------------


def slim(results):
    """The figures do not need every digit of every history row."""

    def tidy(value):
        if isinstance(value, float):
            return float(f"{value:.5g}")
        if isinstance(value, list):
            return [tidy(v) for v in value]
        if isinstance(value, dict):
            return {k: tidy(v) for k, v in value.items()}
        return value

    return "window.RESULTS = " + json.dumps(tidy(results), separators=(",", ":")) + ";\n"


def main():
    check = "--check" in sys.argv
    results = load_results()
    out = build_pages(numbers(results), repo_url())
    source = build_source()
    check_links(out, source)
    out["assets/source-data.js"] = source
    out["assets/results-data.js"] = slim(results)
    out[".nojekyll"] = ""

    stale = []
    for name, text in out.items():
        path = DOCS / name
        if path.exists() and path.read_text(encoding="utf-8") == text:
            continue
        # The log lists commits, so it always changes one commit later than it
        # is written. The repository link depends on the local remote.
        if check and (name == "log.html" or name.endswith(".html") and _same_but_repo(path, text)):
            continue
        stale.append(name)
        if not check:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8", newline="\n")
    if check and stale:
        raise SystemExit("out of date, run python -m tools.guide.build: " + ", ".join(stale))
    print("up to date" if check else f"wrote {len(stale)} files")


def _same_but_repo(path, text):
    def strip(s):
        return re.sub(r'<a href="https://github\.com/[^"]+">Repository</a>', "", s)

    return path.exists() and strip(path.read_text(encoding="utf-8")) == strip(text)


if __name__ == "__main__":
    main()
