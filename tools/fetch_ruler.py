#!/usr/bin/env python3
"""Fetch RULER's essays and words, and log what was fetched (#208).

    .venv/bin/python tools/fetch_ruler.py --issue 208

As RULER's `scripts/data/synthetic/json/download_paulgraham_essay.py`
fetches them: every URL of its `PaulGrahamEssays_URLs.txt`, at a pinned
RULER commit, the GitHub copies of gkamradt's needle-in-a-haystack repo
and paulgraham.com's pages, joined into one text (ruler.essay_text); and
wonderwords' adjective and noun lists, at a pinned release, RULER's words.
The standard library does the fetching and the pages' extraction, not
html2text (ADR-0017's amendment). A URL that fails is skipped and named,
as RULER's script skips it.

Writes data/ruler/ (ruler.DATA): PaulGrahamEssays.json, {"text": ...} as
RULER's, the two word lists, the URL list, and every page as fetched, in
fetched/; and logs a "ruler-data"
entry naming each file's sha256, what every RULER run is checked against.
Refuses a dirty tree, and a data/ruler/ that holds the files already.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from microinfer import benchlog, ruler  # noqa: E402

#: RULER's commit whose URL list is fetched (main on 2026-11-15).
RULER_COMMIT = "db43f92ff634bbce1c33d1fb15858aec4bba509a"
URL_LIST = (f"https://raw.githubusercontent.com/NVIDIA/RULER/{RULER_COMMIT}/scripts/data/"
            f"synthetic/json/PaulGrahamEssays_URLs.txt")
#: wonderwords' last release, whose lists RULER's niah.py reads.
WONDERWORDS = "v2.2.0"
WORD_LISTS = {name: (f"https://raw.githubusercontent.com/mrmaxguns/wonderwordsmodule/"
                     f"{WONDERWORDS}/wonderwords/assets/{name}")
              for name in (ruler.ADJECTIVES, ruler.NOUNS)}
#: A client of no one in particular, and a pause between requests.
USER_AGENT = "microinfer-data-fetch/1.0"
PAUSE_S = 0.3


def get(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8", errors="replace")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--issue", type=int, required=True)
    parser.add_argument("--out", type=Path, default=ruler.DATA)
    parser.add_argument("--log", type=Path, default=benchlog.DEFAULT_LOG)
    args = parser.parse_args(argv)
    if benchlog.environment(args.log)["git_dirty"]:
        parser.error("tracked files have uncommitted changes; commit first, so that the "
                     "entry names the code that produced it")
    if (args.out / ruler.ESSAYS).exists():
        parser.error(f"{args.out / ruler.ESSAYS} exists; the data is fetched once, and its "
                     f"hashes are what every run is checked against")
    args.out.mkdir(parents=True, exist_ok=True)

    url_list = get(URL_LIST)
    urls = [line.strip() for line in url_list.splitlines() if line.strip()]
    fetched, failed = {}, {}
    for i, url in enumerate(urls, 1):
        try:
            fetched[url] = get(url)
        except Exception as exc:  # noqa: BLE001 - RULER's script skips a failure too
            failed[url] = str(exc)
        print(f"{i}/{len(urls)} {'FAILED ' if url in failed else ''}{url}", flush=True)
        time.sleep(PAUSE_S)
    text = ruler.essay_text(fetched)

    # Each page as fetched, so that its text can be extracted again offline.
    raw = args.out / "fetched"
    raw.mkdir(exist_ok=True)
    for url, content in fetched.items():
        (raw / url.split("/")[-1]).write_text(content)
    (args.out / "PaulGrahamEssays_URLs.txt").write_text(url_list)
    (args.out / ruler.ESSAYS).write_text(json.dumps({"text": text}))
    for name, url in WORD_LISTS.items():
        (args.out / name).write_text(get(url))
    files = [ruler.ESSAYS, ruler.ADJECTIVES, ruler.NOUNS, "PaulGrahamEssays_URLs.txt"]
    results = {"files": {name: ruler.sha256(args.out / name) for name in files},
               "urls": len(urls), "fetched": len(fetched), "failed": failed,
               "github_copies": sum(".html" not in u for u in fetched),
               "pages": sum(".html" in u for u in fetched),
               "characters": len(text), "words": len(text.split())}
    benchlog.append("ruler-data", model=None, context_length=None, precision_tiers=None,
                    config={"issue": args.issue, "ruler_commit": RULER_COMMIT,
                            "url_list": URL_LIST, "wonderwords": WONDERWORDS,
                            "word_lists": WORD_LISTS, "extraction": "ruler.font_text"},
                    results=results, log=args.log)
    print(json.dumps({k: v for k, v in results.items() if k != "failed"}, indent=1))
    print(f"{len(failed)} failed: {sorted(failed)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
