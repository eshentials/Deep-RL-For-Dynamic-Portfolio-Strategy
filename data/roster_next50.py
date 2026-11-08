"""Point-in-time Nifty Next 50 roster (protocol v2).

Evidence, in the order it is used:
1. niftyindices.com press releases. The site's press-release page lists every notice with
   its date and title; the index-change notices ("Change in Indices w.e.f. ...",
   "Replacement(s) in indices ...") are downloaded directly (plain requests, no bot wall)
   and their text extracted with `pdftotext -layout`. The PDFs stay in an ignored cache;
   the extracted text and each PDF's URL and SHA-256 are committed.
2. Archived copies of NSE's own constituent files (Wayback Machine id_ captures):
   - Next 50 lists: ind_jrniftylist.csv (CNX Nifty Junior, 2013-2016) and
     ind_niftynext50list.csv (2018 onward, niftyindices and NSE archive hosts);
   - Nifty 100 lists: ind_cnx100list.csv and ind_nifty100list.csv. The Next 50 is the
     Nifty 100 minus the Nifty 50, so a Nifty 100 copy minus protocol v1's validated
     point-in-time Nifty 50 on that date is a Next 50 reading too.
   Every copy is a validation point for the roster built from the dated changes.

Run (always under the v2 protocol, so outputs land in studies/next50/):
  RB_PROTOCOL=protocol_v2.yaml python -m data.roster_next50 notices     # listing, PDFs, text
  RB_PROTOCOL=protocol_v2.yaml python -m data.roster_next50 sections    # Next 50 passages per notice
  RB_PROTOCOL=protocol_v2.yaml python -m data.roster_next50 snapshots   # archived lists -> snapshot tables
  RB_PROTOCOL=protocol_v2.yaml python -m data.roster_next50 build       # changes.csv -> daily membership + validation
"""
from __future__ import annotations

import hashlib
import html
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import requests

from config import ROOT, load, study_root

if load()["data"]["universe"] != "next50":
    sys.exit("data.roster_next50 runs under the v2 protocol: set RB_PROTOCOL=protocol_v2.yaml")

BASE = study_root() / "data"
RAW = BASE / "raw"
NOTICE_DIR = RAW / "press_releases"          # committed: listing.csv, notices.csv, text/*.txt
PDF_CACHE = ROOT / "scratch" / "press_release_pdfs"   # ignored (scratch/)
SNAP_DIR = RAW / "roster_snapshots"
OUT = BASE / "roster"

UA = {"User-Agent": "research-data-builder/1.0 (point-in-time index membership study)"}
SITE = "https://www.niftyindices.com"
LISTING_URL = f"{SITE}/press-release"
PAUSE = 2.5
FIRST, LAST = "2013-10-01", "2026-11-06"

# Every equity notice in the window is downloaded and its text decides relevance (a title
# filter missed "Deferment of Index Rebalancing" and "Exclusion of Yes Bank Ltd. ..." in 2020).
# Left out: bond and debt indices, the SME index, launches, licensing and other news.
NOT_EQUITY = re.compile(r"bond|fixed income|g-?sec|gilt|debt|t-bill|sdl|corporate|money market|treasury|overnight"
                        r"|sme emerge|\blaunch|licen|\betf|aum\b|webinar|award|mou\b|partner|tie[- ]?up|conference"
                        r"|seminar|memorandum", re.I)


def _get(url: str, tries: int = 5) -> requests.Response:
    last = None
    for k in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=90)
            if r.status_code == 200:
                return r
            last = f"HTTP {r.status_code}"
            if r.status_code == 404:
                break
        except requests.RequestException as e:
            last = repr(e)[:160]
        time.sleep(20 * (k + 1))
    raise RuntimeError(f"GET failed: {url} ({last})")


def listing() -> pd.DataFrame:
    """Every press release on niftyindices.com's press-release page (date, file, title)."""
    r = _get(LISTING_URL)
    items = re.findall(r'data-date="([^"]+)"[^>]*>\s*<p>[^<]*</p>\s*<a href=\'([^\']+)\'[^>]*>(.*?)</a>',
                       r.text, flags=re.S)
    df = pd.DataFrame([(pd.to_datetime(d).date(), h.split("/")[-1], html.unescape(re.sub(r"\s+", " ", t)).strip())
                       for d, h, t in items], columns=["date", "file", "title"]).drop_duplicates(["file", "title"])
    NOTICE_DIR.mkdir(parents=True, exist_ok=True)
    df.sort_values(["date", "file"]).to_csv(NOTICE_DIR / "listing.csv", index=False)
    meta = {"url": LISTING_URL, "fetched_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "html_sha256": hashlib.sha256(r.content).hexdigest(), "items": len(df)}
    pd.Series(meta).to_csv(NOTICE_DIR / "listing_meta.csv", header=False)
    return df


def candidates(df: pd.DataFrame) -> pd.DataFrame:
    d = pd.to_datetime(df["date"])
    keep = (d >= FIRST) & (d <= LAST) & ~df["title"].str.contains(NOT_EQUITY)
    return df[keep].sort_values(["date", "file"]).reset_index(drop=True)


def ocr(pdf, txt) -> None:
    """Image-only notice: 300 dpi pages through Tesseract; the text is marked as OCR output."""
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["pdftoppm", "-r", "300", "-png", str(pdf), f"{tmp}/p"], check=True)
        pages = []
        for png in sorted(Path(tmp).glob("p*.png")):
            out = subprocess.run(["tesseract", str(png), "-", "--psm", "6"], capture_output=True, text=True,
                                 check=True).stdout
            pages.append(out)
    txt.write_text("[OCR: tesseract --psm 6 on 300 dpi pages; check symbols against the PDF]\n\n"
                   + "\n\f".join(pages))


def notices(refresh_listing: bool = False) -> pd.DataFrame:
    """Download each candidate notice, hash it, extract its text."""
    lp = NOTICE_DIR / "listing.csv"
    df = listing() if refresh_listing or not lp.exists() else pd.read_csv(lp)
    cand = candidates(df)
    (NOTICE_DIR / "text").mkdir(parents=True, exist_ok=True)
    PDF_CACHE.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in cand.itertuples():
        url = f"{SITE}/Press_Release/{r.file}"
        pdf = PDF_CACHE / r.file

        def failed(msg):
            rows.append({"date": r.date, "file": r.file, "title": r.title, "url": url, "pdf_sha256": "",
                         "bytes": 0, "text_file": "", "error": msg})
        if not pdf.exists():
            try:
                content = _get(url).content
            except RuntimeError as e:
                failed(str(e))
                continue
            time.sleep(PAUSE)
            if not content.startswith(b"%PDF"):   # the site answers some missing files with an HTML page
                failed("not a PDF: the site returned an HTML page")
                continue
            pdf.write_bytes(content)
        elif not pdf.read_bytes()[:4] == b"%PDF":
            pdf.unlink()
            failed("not a PDF: the site returned an HTML page")
            continue
        txt = NOTICE_DIR / "text" / (pdf.stem + ".txt")
        if not txt.exists():
            done = subprocess.run(["pdftotext", "-layout", str(pdf), str(txt)], capture_output=True)
            if done.returncode != 0:
                txt.write_text("")
        if len(txt.read_text(errors="replace").strip()) < 100:  # image-only PDF
            ocr(pdf, txt)
        rows.append({"date": r.date, "file": r.file, "title": r.title, "url": url,
                     "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(), "bytes": pdf.stat().st_size,
                     "text_file": str(txt.relative_to(ROOT)), "error": ""})
        print(f"{r.date} {r.file:24s} {r.title[:80]}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(NOTICE_DIR / "notices.csv", index=False)
    return out


# ---------------------------------------------------------------- Next 50 passages in the notices

NEXT50_NAME = re.compile(r"\b(?:cnx\s+nifty\s+junior|nifty\s+junior|nifty\s*next\s*50)\b", re.I)
# A section heading always carries an enumerator ("1.", "(2)", "7)", "d)", "ii."); without that
# requirement a wrapped note line that happens to start with "Nifty 50 index" closes the section.
INDEX_HEADER = re.compile(r"^\s*\(?(?:\d{1,2}|[ivx]{1,5}|[a-h])[\).]\s+(?:cnx|nifty|s&p\s+cnx)\b[^:]{0,60}:?$", re.I)
ROW = re.compile(r"^\s*(\d{1,3})\s+(.+?)\s{2,}(\S.*?)\s*$")
WEF = re.compile(r"effective\s+from\s+(\w+\s+\d{1,2},?\s+\d{4})|w\.?\s*e\.?\s*f\.?\s+(\w+\s+\d{1,2},?\s+\d{4})", re.I)


def _effective_dates(text: str, title: str) -> list[str]:
    out = []
    for m in WEF.finditer(title + "\n" + text):
        raw = (m.group(1) or m.group(2)).replace(",", "")
        try:
            out.append(pd.to_datetime(raw, format="%B %d %Y").date().isoformat())
        except ValueError:
            pass
    return sorted(set(out))


def parse_next50(text: str) -> list[dict]:
    """Rows of the Next 50 section(s): action (exclude/include), company, symbol, line number."""
    rows, in_n50, mode = [], False, None
    for i, line in enumerate(text.splitlines(), start=1):
        t = line.strip()
        if not t:
            continue
        if INDEX_HEADER.match(t) and "following" not in t.lower():
            in_n50, mode = bool(NEXT50_NAME.search(t)), None
            continue
        low = t.lower()
        if "being excluded" in low or "exclusion" in low and "following" in low:
            mode = "exclude"
            if NEXT50_NAME.search(t):
                in_n50 = True
            continue
        if "being included" in low or "inclusion" in low and "following" in low:
            mode = "include"
            if NEXT50_NAME.search(t):
                in_n50 = True
            continue
        if in_n50 and mode:
            m = ROW.match(line)
            if m:
                rows.append({"action": mode, "company": m.group(2).strip(), "symbol": m.group(3).strip(),
                             "line": i})
    return rows


def sections() -> pd.DataFrame:
    """Auto-parsed Next 50 changes per notice, plus every notice that mentions the index,
    for the hand check that produces data/roster/changes.csv."""
    nt = pd.read_csv(NOTICE_DIR / "notices.csv")
    out = []
    for r in nt[nt["error"].isna() | (nt["error"] == "")].itertuples():
        text = (ROOT / r.text_file).read_text(errors="replace")
        mentions = len(NEXT50_NAME.findall(text))
        parsed = parse_next50(text)
        eff = " ".join(_effective_dates(text, r.title))
        if not mentions and not parsed:
            continue
        if not parsed:
            out.append({"date": r.date, "file": r.file, "effective": eff, "action": "", "company": "",
                        "symbol": "", "line": "", "mentions": mentions, "title": r.title})
        for x in parsed:
            out.append({"date": r.date, "file": r.file, "effective": eff, **x, "mentions": mentions,
                        "title": r.title})
    df = pd.DataFrame(out)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "notice_rows_auto.csv", index=False)
    return df


def review_queue() -> pd.DataFrame:
    """Notices that need a hand read: index mentioned but nothing parsed, unequal exclusion and
    inclusion counts, several effective dates, or OCR text."""
    a = pd.read_csv(OUT / "notice_rows_auto.csv", dtype=str).fillna("")
    q = []
    for f, g in a.groupby("file", sort=False):
        n_ex, n_in = int((g["action"] == "exclude").sum()), int((g["action"] == "include").sum())
        text = (NOTICE_DIR / "text" / f.replace(".pdf", ".txt")).read_text(errors="replace")
        why = []
        if n_ex + n_in == 0:
            why.append("mentioned, nothing parsed")
        if n_ex != n_in:
            why.append(f"{n_ex} out / {n_in} in")
        if len(g["effective"].iloc[0].split()) > 1:
            why.append("several effective dates")
        if text.startswith("[OCR"):
            why.append("OCR text")
        if why:
            q.append({"date": g["date"].iloc[0], "file": f, "why": "; ".join(why), "title": g["title"].iloc[0]})
    return pd.DataFrame(q)


# ---------------------------------------------------------------- hand decisions (each one documented)

SKIP_NOTICES = {
    "ind_prs18022020.pdf": "March 2020 review deferred 'until further notice' (ind_prs23032020.pdf, COVID-19) and "
                           "redone with different lists effective 2020-06-26 (ind_prs10062020.pdf)",
}
EFFECTIVE_OVERRIDE = {
    "ind_prs17102016.pdf": ("2016-11-15", "Next 50 change is in part C, effective November 15, 2016; "
                                          "October 24 applies to the Alpha/Beta/Low Vol indices"),
    "ind_prs16032020.pdf": ("2020-03-19", "Yes Bank removal brought forward to March 19; March 27 refers to the "
                                          "review later deferred"),
    "ind_prs02072020_1.pdf": ("2020-07-31", "July 9 applies to the SME EMERGE index only"),
}
# Image-only notice (no text layer): rows read by hand from the rendered PDF pages 2-3.
MANUAL_ROWS = [("ind_prs23082021.pdf", "2021-09-30", "remove", s) for s in
               ("ABBOTINDIA", "ALKEM", "MRF", "PETRONET", "UBL")] + \
              [("ind_prs23082021.pdf", "2021-09-30", "add", s) for s in
               ("BANKBARODA", "CHOLAFIN", "JINDALSTEL", "PIIND", "SAIL")]
AD_HOC = {
    "ind_prs18112014.pdf": "ad hoc: Ranbaxy amalgamation with Sun Pharma",
    "ind_prs21012015.pdf": "ad hoc: ING Vysya Bank amalgamation with Kotak Mahindra Bank",
    "ind_prs29042015.pdf": "ad hoc: Bosch to CNX Nifty in place of IDFC; Adani Enterprises replaced",
    "ind_prs24082015.pdf": "ad hoc: Crompton Greaves demerger (effective with the review date)",
    "ind_prs18092015.pdf": "ad hoc: Aditya Birla Nuvo demerger of the Madura undertaking",
    "ind_prs17102016.pdf": "ad hoc: Cairn India amalgamation into Vedanta",
    "ind_prs27042017.pdf": "ad hoc: Aditya Birla Nuvo merger with Grasim; Vedanta to Nifty 50",
    "ind_prs16032020.pdf": "ad hoc: Yes Bank removal brought forward; Shree Cement to Nifty 50",
    "ind_prs02072020_1.pdf": "ad hoc: Vedanta proposed delisting; HDFC Life to Nifty 50",
    "ind_prs15062021.pdf": "ad hoc: Motherson Sumi demerger of the domestic wiring harness business",
    "ind_prs11072022.pdf": "ad hoc: NMDC and Piramal Enterprises demergers",
    "ind_prs04072023.pdf": "ad hoc: HDFC amalgamation with HDFC Bank; LTIMindtree to Nifty 50",
}
REVIEW_NOTE = {
    "ind_prs10062020.pdf": "semi-annual review (March 2020 review deferred, COVID-19)",
    "ind_prs23082021.pdf": "semi-annual review (image-only notice; rows read from the rendered pages)",
}


def build_changes() -> pd.DataFrame:
    """data/roster/changes.csv: parsed notice rows plus the documented hand decisions above."""
    a = pd.read_csv(OUT / "notice_rows_auto.csv", dtype=str).fillna("")
    a = a[(a["action"] != "") & ~a["file"].isin(SKIP_NOTICES)]
    rows = []
    for r in a.itertuples():
        eff = r.effective.split()
        if r.file in EFFECTIVE_OVERRIDE:
            eff = [EFFECTIVE_OVERRIDE[r.file][0]]
        if len(eff) != 1:
            raise ValueError(f"{r.file}: effective date needs a decision ({r.effective})")
        rows.append({"effective_date": eff[0], "action": {"exclude": "remove", "include": "add"}[r.action],
                     "symbol": r.symbol, "notice": r.file, "notice_date": r.date})
    nd = pd.read_csv(NOTICE_DIR / "notices.csv", dtype=str).set_index("file")["date"]
    for f, eff, act, sym in MANUAL_ROWS:
        rows.append({"effective_date": eff, "action": act, "symbol": sym, "notice": f, "notice_date": nd[f]})
    ch = pd.DataFrame(rows)
    ch["reason"] = [AD_HOC.get(f) or REVIEW_NOTE.get(f) or "semi-annual review" for f in ch["notice"]]
    ch["confidence"] = "high"     # every row rests on an NSE Indices notice
    ch["source"] = SITE + "/Press_Release/" + ch["notice"]
    ch = ch.sort_values(["effective_date", "action", "symbol"], ascending=[True, False, True])
    cols = ["effective_date", "action", "symbol", "reason", "confidence", "source", "notice_date"]
    ch[cols].to_csv(OUT / "changes.csv", index=False)
    pd.DataFrame([{"notice": k, "decision": "skipped", "why": v} for k, v in SKIP_NOTICES.items()]
                 + [{"notice": k, "decision": f"effective {v[0]}", "why": v[1]} for k, v in EFFECTIVE_OVERRIDE.items()]
                 + [{"notice": "ind_prs23082021.pdf", "decision": "rows entered by hand",
                     "why": "image-only PDF; OCR garbled the tables; read from the rendered pages"}]
                 ).to_csv(OUT / "hand_decisions.csv", index=False)
    return ch[cols]


def draft_changes() -> pd.DataFrame:
    """First draft of data/roster/changes.csv from the parsed rows (hand-checked afterwards)."""
    a = pd.read_csv(OUT / "notice_rows_auto.csv", dtype=str).fillna("")
    a = a[a["action"] != ""]
    rows = []
    for r in a.itertuples():
        eff = r.effective.split()
        rows.append({"effective_date": eff[0] if len(eff) == 1 else "CHECK:" + "|".join(eff),
                     "action": {"exclude": "remove", "include": "add"}[r.action], "symbol": r.symbol,
                     "reason": "", "confidence": "high", "source": f"{SITE}/Press_Release/{r.file}"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- archived constituent lists

from data.roster import Source, cdx_captures, parse_csv  # noqa: E402
from data.roster import _get as _archive_get  # noqa: E402

SNAPSHOT_SOURCES = [
    # (source, list): "next50" lists are direct readings; "nifty100" lists give Next 50 = Nifty 100 - Nifty 50
    (Source("www.nseindia.com/content/indices/ind_jrniftylist.csv", "csv"), "next50"),
    (Source("www.niftyindices.com/IndexConstituent/ind_niftynext50list.csv", "csv"), "next50"),
    (Source("archives.nseindia.com/content/indices/ind_niftynext50list.csv", "csv"), "next50"),
    (Source("nsearchives.nseindia.com/content/indices/ind_niftynext50list.csv", "csv"), "next50"),
    (Source("www.nseindia.com/content/indices/ind_cnx100list.csv", "csv"), "nifty100"),
    (Source("www.nseindia.com/content/indices/ind_nifty100list.csv", "csv"), "nifty100"),
    (Source("archives.nseindia.com/content/indices/ind_nifty100list.csv", "csv"), "nifty100"),
    (Source("www.niftyindices.com/IndexConstituent/ind_nifty100list.csv", "csv"), "nifty100"),
]


def _snap_path(ts: str, original: str) -> "Path":
    host = original.split("//", 1)[-1].split("/", 1)[0].split(":")[0]
    stem = original.rsplit("/", 1)[-1].split("?")[0].replace(".csv", "")
    return SNAP_DIR / f"{ts}_{host}_{stem}.csv"


def served_capture(ts: str, original: str) -> bytes:
    """Bytes of exactly the capture asked for. The archive has once answered a request for one
    capture with a later one (a 2024 timestamp served the 2026-06-21 file), so the served
    Memento-Datetime must equal the requested timestamp, with no redirect followed."""
    url = f"https://web.archive.org/web/{ts}id_/{original}"
    last = None
    for k in range(6):
        try:
            r = requests.get(url, headers=UA, timeout=90, allow_redirects=False)
            served = r.headers.get("Memento-Datetime")
            if r.status_code == 200 and served and pd.Timestamp(served).strftime("%Y%m%d%H%M%S") == ts:
                return r.content
            last = f"HTTP {r.status_code}, served {served}"
        except requests.RequestException as e:   # a refused connection is the archive's rate limit
            last = repr(e)[:120]
        time.sleep(45 * (k + 1))
    raise RuntimeError(f"archive did not serve capture {ts} of {original} ({last})")


def verify_snapshots() -> pd.DataFrame:
    """Re-fetch every cached archive copy under the Memento guard and compare bytes."""
    caps = pd.read_csv(SNAP_DIR / "cdx_captures.csv", dtype=str)
    caps = caps[caps["statuscode"] == "200"].drop_duplicates(["timestamp", "original"])
    rows = []
    for c in caps.itertuples():
        path = _snap_path(c.timestamp, c.original)
        if not path.exists():
            continue
        fresh = served_capture(c.timestamp, c.original)
        rows.append({"capture_ts": c.timestamp, "file": path.name, "identical": fresh == path.read_bytes()})
        print(rows[-1], flush=True)
        time.sleep(10)
    return pd.DataFrame(rows)


def snapshots() -> pd.DataFrame:
    """Download and parse every archived copy; one row per parsed capture."""
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    cache = SNAP_DIR / "cdx_captures.csv"
    if cache.exists():
        caps = pd.read_csv(cache, dtype=str)
    else:
        parts = []
        for src, lst in SNAPSHOT_SOURCES:
            c = cdx_captures(src)
            c["list"] = lst
            parts.append(c)
            time.sleep(5)
        caps = pd.concat(parts, ignore_index=True)
        caps.to_csv(cache, index=False)
    caps = caps[caps["statuscode"] == "200"].drop_duplicates(["timestamp", "original"])
    index_rows, member_rows, parsed = [], [], {}
    for c in caps.sort_values("timestamp").itertuples():
        key = (c.digest, c.list)
        if key not in parsed:
            path = _snap_path(c.timestamp, c.original)
            if not path.exists():
                path.write_bytes(served_capture(c.timestamp, c.original))
                time.sleep(5)
            parsed[key] = parse_csv(path.read_bytes())
        df = parsed[key]
        if df is None or df.empty:
            continue
        syms = sorted(df["symbol"])
        mh = hashlib.sha256(",".join(syms).encode()).hexdigest()[:12]
        index_rows.append({"capture_ts": c.timestamp, "date": pd.Timestamp(c.timestamp[:8]).date(),
                           "source": c.source, "list": c.list, "digest": c.digest, "n_members": len(syms),
                           "members_hash": mh})
        for m in df.itertuples(index=False):
            member_rows.append({"capture_ts": c.timestamp, "list": c.list, "members_hash": mh,
                                "symbol": m.symbol, "isin": m.isin, "company": m.company, "industry": m.industry})
    idx = pd.DataFrame(index_rows).sort_values(["date", "capture_ts"]).reset_index(drop=True)
    pd.DataFrame(member_rows).to_csv(OUT / "snapshot_members.csv", index=False)
    idx.to_csv(OUT / "snapshot_index.csv", index=False)
    return idx


# ---------------------------------------------------------------- build and validate

V1_ROSTER = ROOT / "data" / "roster"          # protocol v1's validated Nifty 50 (read only)
ANCHOR_CAPTURE = "20140122091423"             # CNX 100 copy of 2014-01-22, minus the Nifty 50 of that day

# Demerger placeholders: in the index from the parent's ex-date until NSE removed the listed
# entity (dates from the notices). Never part of the roster, as in protocol v1; dropped from
# archived readings inside these windows. (first session, removal effective, notice)
PLACEHOLDERS = {
    "ENRIN": ("2025-04-07", "2025-06-27", "ind_prs25062025_2.pdf"),   # Siemens Energy India (DUMMYSIEMS)
    "VAML": ("2026-04-30", "2026-06-23", "ind_prs19062026.pdf"),      # Vedanta Aluminium Metal (DUMMYVEDL*)
    "VOGL": ("2026-04-30", "2026-06-24", "ind_prs22062026.pdf"),      # Vedanta Oil and Gas
    "VPL": ("2026-04-30", "2026-06-19", "ind_prs17062026.pdf"),       # Vedanta Power
    "VISL": ("2026-04-30", "2026-06-19", "ind_prs17062026.pdf"),      # Vedanta Iron and Steel
}
# Tata Motors' DVR share stayed in the Nifty 100 as an additional security of Tata Motors (a
# Nifty 50 member) after leaving the Nifty 50 on 2017-09-29, so Nifty 100 copies of 2017-2019
# list 101 lines; no direct Next 50 copy ever lists it. Dropped from Nifty 100-derived readings.
NIFTY100_EXTRA_SECURITIES = {"TATAMTRDVR"}

# Documented windows in which the index does not hold exactly 50 securities. Filled in
# from the notices during the hand check (demerger placeholders and the like).
COUNT_EXCEPTIONS: list[tuple[str, str, int, str]] = []   # (first session, last session, count, reason)
KNOWN_STALE: dict[str, str] = {}                          # capture_ts -> why that copy is not point in time


def canonical_map() -> dict[str, str]:
    """symbol -> latest symbol NSE printed for the same equity ISIN (both studies' lists)."""
    parts = []
    for root in (OUT, V1_ROSTER):
        m = pd.read_csv(root / "snapshot_members.csv", dtype=str)
        idx = pd.read_csv(root / "snapshot_index.csv", dtype={"capture_ts": str}, parse_dates=["date"])
        parts.append(m.merge(idx[["capture_ts", "date"]].drop_duplicates("capture_ts"), on="capture_ts"))
    m = pd.concat(parts)
    m = m[(m["isin"].str.len() == 12) & (m["isin"].str[7:9] == "01")].sort_values("date")
    latest = m.groupby("isin")["symbol"].last()
    return {s: latest[i] for s, i in m[["symbol", "isin"]].drop_duplicates().itertuples(index=False)}


def _members_all() -> pd.DataFrame:
    return pd.concat([pd.read_csv(r / "snapshot_members.csv", dtype=str) for r in (OUT, V1_ROSTER)])


def isin_of(canon: dict[str, str]) -> dict[str, str]:
    """canonical symbol -> its equity ISIN as printed in NSE's lists."""
    m = _members_all()
    m = m[(m["isin"].str.len() == 12) & (m["isin"].str[7:9] == "01")]
    return {canon.get(s, s): i for s, i in m[["symbol", "isin"]].drop_duplicates().itertuples(index=False)}


def aliases_table(canon: dict[str, str]) -> pd.DataFrame:
    """Every symbol NSE used for each canonical symbol (lists and notices), for exchange-file lookups."""
    syms = set(_members_all()["symbol"]) | set(pd.read_csv(OUT / "notice_rows_auto.csv", dtype=str)["symbol"].dropna())
    rows = [{"symbol": canon.get(s, s), "alias": s} for s in sorted(syms)]
    return pd.DataFrame(rows).drop_duplicates().sort_values(["symbol", "alias"])


def sessions() -> pd.DatetimeIndex:
    cfg = load()["data"]
    s = pd.read_parquet(BASE / "processed" / "market.parquet").index
    return s[(s >= pd.Timestamp(cfg["study_start"])) & (s <= pd.Timestamp(cfg["freeze"]))]


def readings(canon: dict[str, str]) -> pd.DataFrame:
    """Every archived Next 50 reading: direct lists, and Nifty 100 lists minus the v1 Nifty 50."""
    idx = pd.read_csv(OUT / "snapshot_index.csv", dtype={"capture_ts": str}, parse_dates=["date"])
    mem = pd.read_csv(OUT / "snapshot_members.csv", dtype=str)
    n50 = pd.read_parquet(V1_ROSTER / "membership_daily.parquet")
    rows = []
    # Demerger placeholders (ISIN prefix "DUM", e.g. DUMMYREL for Jio Financial in Aug 2023) carry
    # no tradable price and are excluded, as in protocol v1's roster.
    mem = mem[~mem["isin"].fillna("").str.startswith("DUM") & ~mem["symbol"].str.startswith("DUMMY")]
    for r in idx.drop_duplicates(["date", "members_hash", "list"]).itertuples():
        syms = {canon.get(x, x) for x in mem.loc[(mem["capture_ts"] == r.capture_ts), "symbol"]}
        if r.list == "nifty100":
            on = n50.index[n50.index <= r.date]
            if not len(on):
                continue
            nifty50 = {canon.get(x, x) for x in n50.columns[n50.loc[on[-1]]]}
            syms = syms - nifty50
        if r.list == "nifty100":
            syms = syms - NIFTY100_EXTRA_SECURITIES
        for sym, (a, b, _) in PLACEHOLDERS.items():
            if pd.Timestamp(a) <= r.date < pd.Timestamp(b):
                syms.discard(sym)
        rows.append({"capture_ts": r.capture_ts, "date": r.date, "source": r.source, "list": r.list,
                     "members": syms})
    return pd.DataFrame(rows)


def build() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    canon = canonical_map()
    rd = readings(canon)
    anchor = rd.loc[rd["capture_ts"] == ANCHOR_CAPTURE, "members"].iloc[0]
    ch = pd.read_csv(OUT / "changes.csv", parse_dates=["effective_date"])
    ch = ch[ch["action"].isin(["add", "remove"])].copy()
    ch["symbol"] = ch["symbol"].map(lambda x: canon.get(x, x))
    S = sessions()
    if (ch["effective_date"] <= pd.Timestamp(rd.loc[rd["capture_ts"] == ANCHOR_CAPTURE, "date"].iloc[0])).any():
        raise ValueError("a change takes effect before the anchor reading")

    current, open_at, intervals = set(anchor), {x: S[0] for x in anchor}, []
    for d, g in ch.groupby("effective_date", sort=True):
        eff = S[S >= d][0]
        for x in g.loc[g["action"] == "remove", "symbol"]:
            if x not in current:
                raise ValueError(f"{d.date()}: removing non-member {x}")
            intervals.append((x, open_at.pop(x), eff))
            current.remove(x)
        for x in g.loc[g["action"] == "add", "symbol"]:
            if x in current:
                raise ValueError(f"{d.date()}: adding existing member {x}")
            current.add(x)
            open_at[x] = eff
    end = S[-1] + pd.Timedelta(days=1)
    intervals += [(x, t0, end) for x, t0 in open_at.items()]
    iv = pd.DataFrame(intervals, columns=["symbol", "start", "end_exclusive"]).sort_values(["symbol", "start"])
    iv = iv.reset_index(drop=True)
    daily = pd.DataFrame(False, index=S, columns=sorted(iv["symbol"].unique()))
    for r in iv.itertuples():
        daily.loc[(S >= r.start) & (S < r.end_exclusive), r.symbol] = True

    rows = []
    for r in rd.itertuples():
        if r.date < S[0]:
            continue
        on = S[S <= r.date][-1]
        built = set(daily.columns[daily.loc[on]])
        ok = built == r.members
        rows.append({"capture_ts": r.capture_ts, "snapshot_date": r.date.date(), "session": on.date(),
                     "source": r.source, "list": r.list, "match": ok,
                     "missing_in_build": " ".join(sorted(r.members - built)),
                     "extra_in_build": " ".join(sorted(built - r.members)),
                     "explanation": "" if ok else KNOWN_STALE.get(r.capture_ts, "UNEXPLAINED")})
    val = pd.DataFrame(rows)

    expected = pd.Series(50, index=S)
    for a, b, n, _ in COUNT_EXCEPTIONS:
        expected[(S >= pd.Timestamp(a)) & (S <= pd.Timestamp(b))] = n
    bad = daily.sum(axis=1)[daily.sum(axis=1) != expected]
    n50 = pd.read_parquet(V1_ROSTER / "membership_daily.parquet").reindex(S).fillna(False)
    n50.columns = [canon.get(c, c) for c in n50.columns]
    both = [c for c in daily.columns if c in n50.columns]
    overlap = (daily[both] & n50[both].astype(bool)).sum(axis=0)
    overlap = overlap[overlap > 0]

    aliases_table(canon).to_csv(OUT / "symbol_aliases.csv", index=False)
    iv["isin"] = iv["symbol"].map(isin_of(canon))
    iv.to_csv(OUT / "membership_intervals.csv", index=False)
    daily.to_parquet(OUT / "membership_daily.parquet")
    val.to_csv(OUT / "validation.csv", index=False)
    return iv, val, bad, overlap


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    pd.set_option("display.width", 220, "display.max_colwidth", 120, "display.max_rows", 400)
    if cmd == "notices":
        n = notices(refresh_listing="--refresh" in sys.argv)
        print(f"{len(n)} notices, {int((n['error'] != '').sum())} failed")
    elif cmd == "sections":
        print(sections().to_string(index=False))
        print(review_queue().to_string(index=False))
    elif cmd == "snapshots":
        print(snapshots().to_string())
    elif cmd == "verify":
        v = verify_snapshots()
        print(v.to_string(index=False))
        print(f"{int(v['identical'].sum())}/{len(v)} cached copies identical to the guarded fetch")
    elif cmd == "build":
        ch = build_changes()
        print(f"changes: {len(ch)} rows on {ch['effective_date'].nunique()} dates "
              f"({(ch['action'] == 'add').sum()} adds, {(ch['action'] == 'remove').sum()} removes)")
        iv, val, bad, overlap = build()
        print(f"symbols ever in the Next 50 (study window): {iv['symbol'].nunique()}; intervals {len(iv)}")
        print(f"readings checked: {len(val)}, matching: {int(val['match'].sum())}")
        if (~val["match"]).any():
            print(val[~val["match"]].to_string(index=False))
        print(f"sessions with an unexpected member count: {len(bad)}")
        if len(bad):
            print(bad.groupby((bad.index.to_series().diff() != pd.Timedelta(days=1)).cumsum()).agg(["size"]).head(30))
            print(bad.head(40).to_string())
        print(f"symbols in the Next 50 and the Nifty 50 on the same session: {len(overlap)}")
        if len(overlap):
            print(overlap.to_string())
    else:
        raise SystemExit(__doc__)
