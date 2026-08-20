import sys, os, time, json, csv
import requests
 
UA = {"User-Agent": "schalil@binghamton.edu"}
SUBS = "https://data.sec.gov/submissions/CIK{cik10}.json"
SLEEP = 0.12          # ~8 req/sec, under SEC's 10/sec limit
 
 
def get(url, tries=4):
    """GET with backoff. Returns response or None."""
    for i in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=45)
            if r.status_code == 200:
                return r
            if r.status_code in (429, 500, 502, 503):
                time.sleep(2 ** i)
                continue
            return None
        except requests.RequestException:
            time.sleep(2 ** i)
    return None
 
 
def all_filings(cik10):
    """filings.recent plus the paginated archive files."""
    r = get(SUBS.format(cik10=cik10))
    if r is None:
        return None
    d = r.json()
    time.sleep(SLEEP)
 
    rows = []
    rec = d.get("filings", {}).get("recent", {})
    if rec:
        rows.extend(zip(rec["form"], rec["reportDate"],
                        rec["accessionNumber"], rec["primaryDocument"]))
 
    # older filings live in separate files - large filers need these
    for f in d.get("filings", {}).get("files", []):
        rr = get("https://data.sec.gov/submissions/" + f["name"])
        if rr is None:
            continue
        e = rr.json()
        time.sleep(SLEEP)
        rows.extend(zip(e["form"], e["reportDate"],
                        e["accessionNumber"], e["primaryDocument"]))
    return rows
 
 
def pick_10k(rows, year):
    """
    TNIC year = first 4 digits of datadate, so match on PERIOD OF REPORT,
    not filing date. A June-2023 FYE files in autumn 2023 and is still 2023.
    Prefers plain 10-K over 10-K/A.
    """
    hits = [r for r in rows
            if r[0] in ("10-K", "10-K405", "10-KSB", "10-K/A")
            and r[1][:4] == str(year)]
    if not hits:
        return None
    hits.sort(key=lambda r: (r[0].endswith("/A"), r[1]))
    return hits[0]
 
 
def main(cikfile, year, outdir):
    os.makedirs(outdir, exist_ok=True)
    logpath = f"fetch_log_{year}.csv"
 
    firms = []
    with open(cikfile) as fh:
        for row in csv.DictReader(fh):
            if row.get("cik") in (None, "", "nan"):
                continue
            firms.append((row["gvkey"], str(int(float(row["cik"])))))
 
    print(f"{len(firms)} firms with a CIK")
 
    done = {f.split("_")[0] for f in os.listdir(outdir)}
    log = open(logpath, "a", newline="")
    w = csv.writer(log)
 
    ok = miss = fail = 0
    for i, (gvkey, cik) in enumerate(firms, 1):
        if gvkey in done:
            continue
 
        cik10 = cik.zfill(10)
        rows = all_filings(cik10)
        if rows is None:
            fail += 1
            w.writerow([gvkey, cik, "", "", "submissions_failed"])
            continue
 
        hit = pick_10k(rows, year)
        if hit is None:
            miss += 1
            w.writerow([gvkey, cik, "", "", "no_10k_for_year"])
            continue
 
        form, rdate, adsh, doc = hit
        url = (f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
               f"{adsh.replace('-', '')}/{doc}")
        r = get(url)
        time.sleep(SLEEP)
        if r is None:
            fail += 1
            w.writerow([gvkey, cik, adsh, rdate, "doc_failed"])
            continue
 
        with open(os.path.join(outdir, f"{gvkey}_{cik}_{rdate}.html"),
                  "w", encoding="utf-8") as fo:
            fo.write(r.text)
        ok += 1
        w.writerow([gvkey, cik, adsh, rdate, form])
 
        if i % 100 == 0:
            log.flush()
            print(f"{i}/{len(firms)}  ok={ok} no10k={miss} fail={fail}")
 
    log.close()
    print(f"\nDONE  ok={ok}  no_10k={miss}  failed={fail}")
    print(f"log: {logpath}")
 
 
if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3])