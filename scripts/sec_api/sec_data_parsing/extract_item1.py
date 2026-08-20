"""
Extract Item 1 (Business) text from saved 10-K HTML files.

Usage:
    python extract_item1_v2.py D:/dissertation/data/filings/2023 item1_2023.parquet

Filenames expected as  <gvkey>_<cik>_<reportdate>.html

Fixes over v1:
  - whitespace-tolerant headings. Inline XBRL splits words mid-way, so the
    raw text often reads "ITEM 1. BUSINES S" or "ITEM 1A. RISK FACTOR S".
    Every keyword is now matched with optional whitespace between letters.
  - hard cap at MAX_CHARS. Some filings ran to 600k chars, meaning the
    extraction overran Item 1A into risk factors and MD&A.
  - XML parse warning suppressed.

Requires: beautifulsoup4, lxml, pandas, pyarrow
    pip install beautifulsoup4 lxml pandas pyarrow tqdm
"""

import os
import re
import sys
import glob
import warnings

import pandas as pd
from bs4 import BeautifulSoup

try:
    from bs4 import XMLParsedAsHTMLWarning
    warnings.filterwarnings('ignore', category=XMLParsedAsHTMLWarning)
except ImportError:
    pass

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **k):
        return x


# ------------------------------------------------------------- parameters

MIN_CHARS = 1500          # below this it's a TOC line or cross-reference
MAX_CHARS = 150_000       # above this the extraction has overrun Item 1
FALLBACK_WINDOW = 120_000


# ------------------------------------------------------------- patterns

def flex(word: str) -> str:
    """
    Match a word tolerating whitespace between any two characters.
    'business' -> b\\s*u\\s*s\\s*i\\s*n\\s*e\\s*s\\s*s
    """
    return r'\s*'.join(re.escape(c) for c in word)

# Flexible delimiter allowing any combination of spaces, dots, colons, and dashes
DELIM = r'[\s\.\:\-\u2013\u2014]*'

RE_START = re.compile(
    flex('item') + r's?' + DELIM + r'1' +
    r'(?:' + DELIM + r'and' + DELIM + r'2)?' +
    DELIM + r'(?:the|our)?' + DELIM + flex('business'),
    re.I
)

RE_END_1A = re.compile(
    flex('item') + DELIM + r'1' + DELIM + r'a' + DELIM + flex('risk'),
    re.I
)

RE_END_2 = re.compile(
    flex('item') + DELIM + r'2' + DELIM + flex('propert'),
    re.I
)


# ------------------------------------------------------------- extraction

def html_to_text(path):
    with open(path, encoding='utf-8', errors='ignore') as fh:
        raw = fh.read()
    soup = BeautifulSoup(raw, 'lxml')
    # tables hold financial figures, not product vocabulary
    for tag in soup(['script', 'style']):
        tag.decompose()
    txt = soup.get_text(' ').replace('\xa0', ' ')
    return re.sub(r'\s+', ' ', txt).strip()


def extract_item1(txt):
    """
    Return (body, status).

    Takes the LAST "Item 1 ... Business" heading that still has an Item 1A
    after it. The first such heading is almost always the table of contents.
    """
    starts = [m.start() for m in RE_START.finditer(txt)]
    if not starts:
        return None, 'no_item1_header'

    ends = [m.start() for m in RE_END_1A.finditer(txt)]
    if not ends:
        ends = [m.start() for m in RE_END_2.finditer(txt)]

    if ends:
        valid = [s for s in starts if any(e > s + 200 for e in ends)]
        if not valid:
            return None, 'no_end'
        s = max(valid)
        e = min(x for x in ends if x > s + 200)
    else:
        s = starts[-1]
        e = min(len(txt), s + FALLBACK_WINDOW)

    body = txt[s:e].strip()[:MAX_CHARS]
    if len(body) < MIN_CHARS:
        return None, 'too_short'
    return body, 'ok'


def parse_name(fname):
    stem = os.path.basename(fname).rsplit('.', 1)[0]
    parts = stem.split('_')
    gvkey = int(parts[0])
    cik = int(parts[1]) if len(parts) > 1 else None
    rdate = parts[2] if len(parts) > 2 else None
    return gvkey, cik, rdate


# ------------------------------------------------------------- self-test

def selftest():
    cases = [
        ("ITEM 1. BUSINESS", True),
        ("Item 1. Business", True),
        ("ITEM 1. BUSINES S", True),        # XBRL split
        ("Item 1 - Business", True),
        ("ITEM 1 \u2013 BUSINESS", True),
        ("Item 1: Our Business", True),
        ("Item 2. Properties", False),
    ]
    ok = True
    for text, want in cases:
        got = bool(RE_START.search(text))
        if got != want:
            print(f"  FAIL start {text!r}: got {got}, want {want}")
            ok = False
    for text, want in [("ITEM 1A. RISK FACTORS", True),
                       ("Item 1A. Risk Factors", True),
                       ("ITEM 1A. RISK FACTOR S", True),
                       ("Item 1. Business", False)]:
        got = bool(RE_END_1A.search(text))
        if got != want:
            print(f"  FAIL end {text!r}: got {got}, want {want}")
            ok = False
    print("regex self-test:", "PASS" if ok else "FAIL")
    return ok


# ------------------------------------------------------------- main

def main(indir, outpath):
    if not selftest():
        print("aborting - patterns are broken")
        return

    files = sorted(glob.glob(os.path.join(indir, '*.htm*')))
    print(f"{len(files)} files in {indir}\n")

    rows = []
    for p in tqdm(files):
        gvkey, cik, rdate = parse_name(p)
        try:
            body, status = extract_item1(html_to_text(p))
        except Exception:
            body, status = None, 'parse_error'
        rows.append({'gvkey': gvkey, 'cik': cik, 'rdate': rdate,
                     'status': status,
                     'nchar': len(body) if body else 0,
                     'text': body})

    d = pd.DataFrame(rows)

    print("\n--- status ---")
    print(d.status.value_counts())
    ok = d[d.status == 'ok']
    print(f"\nok rate: {len(ok) / len(d):.1%}")

    print("\n--- length of extracted Item 1 (chars) ---")
    print(ok.nchar.describe(percentiles=[.05, .25, .5, .75, .95]).round(0))

    print(f"\ncapped at {MAX_CHARS:,}: {(ok.nchar >= MAX_CHARS).sum()} filings")

    print("\n--- 5 shortest 'ok' ---")
    print(ok.nsmallest(5, 'nchar')[['gvkey', 'nchar']].to_string(index=False))

    d.to_parquet(outpath, index=False)
    print(f"\nwrote {outpath}")


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])