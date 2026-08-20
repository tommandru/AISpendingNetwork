"""
Hoberg-Phillips style product similarity from extracted Item 1 text,
calibrated against published TNIC scores.

Two modes:

  TAG   - run spaCy once, cache the noun-filtered text. Slow (~90 min).
              python score_similarity.py tag  item1_2023.parquet  nouns_2023.parquet

  SWEEP - load cached nouns, try several parameter settings, report the
          calibration for each. Fast (seconds per setting).
              python score_similarity.py sweep  nouns_2023.parquet  tnicall2023.txt

  ONE   - load cached nouns, run a single setting, write the score file.
              python score_similarity.py one  nouns_2023.parquet  tnicall2023.txt  my_2023.parquet  0.25  5

Method (Hoberg & Phillips 2016 JPE):
  nouns and proper nouns only, geographic terms dropped, BINARY word
  vectors, common words screened out, L2 normalise, cosine.

Requires:
    pip install spacy scikit-learn pandas pyarrow scipy tqdm
    python -m spacy download en_core_web_sm
"""

import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **k):
        return x


# --------------------------------------------------------------- parameters

MAX_DF = 0.25
MIN_DF = 5
FLOOR = 0.0005
BLOCK = 400
TEXT_CAP = 200_000        # chars of Item 1 fed to spaCy
MIN_NOUNS = 50            # firms with fewer usable nouns are dropped


# --------------------------------------------------------------- tagging

def nouns_batch(texts, nlp, batch_size=50, n_process=1):
    GEO = {'GPE', 'LOC', 'FAC', 'NORP'}
    out = []
    for doc in tqdm(nlp.pipe(texts, batch_size=batch_size,
                             n_process=n_process),
                    total=len(texts), desc='POS tagging'):
        geo = {t.i for ent in doc.ents if ent.label_ in GEO for t in ent}
        out.append(' '.join(
            t.text.lower() for t in doc
            if t.pos_ in ('NOUN', 'PROPN')
            and t.i not in geo
            and t.is_alpha
            and len(t.text) > 2
        ))
    return out


def cmd_tag(item1_path, nouns_path, n_process=1):
    """Run spaCy once and cache the result. This is the expensive step."""
    import spacy

    print("loading Item 1 extracts ...")
    d = pd.read_parquet(item1_path)
    d = d[(d.status == 'ok') & d.text.notna()]
    d = d.drop_duplicates('gvkey').reset_index(drop=True)
    print(f"  {len(d):,} firms with usable Item 1")

    nlp = spacy.load('en_core_web_sm', disable=['parser', 'lemmatizer'])
    nlp.max_length = 3_000_000

    texts = [t[:TEXT_CAP] for t in d.text.tolist()]
    d['nouns'] = nouns_batch(texts, nlp, n_process=n_process)

    keep = d.nouns.str.split().str.len() >= MIN_NOUNS
    print(f"  dropping {(~keep).sum()} firms with <{MIN_NOUNS} nouns")
    d = d[keep].reset_index(drop=True)

    d[['gvkey', 'nouns']].to_parquet(nouns_path, index=False)
    print(f"\nwrote {nouns_path}  ({len(d):,} firms)")
    print("tagging is now cached - sweeps are cheap from here")


# --------------------------------------------------------------- similarity

def pairwise_above_floor(X, gvkeys, floor=FLOOR, block=BLOCK, quiet=False):
    """
    Cosine for all pairs, upper triangle only, keeping pairs >= floor.
    X must be L2-normalised so X @ X.T is the cosine matrix.
    """
    n = X.shape[0]
    rows, cols, vals = [], [], []
    it = range(0, n, block)
    if not quiet:
        it = tqdm(it, desc='cosine')

    for start in it:
        stop = min(start + block, n)
        sim = (X[start:stop] @ X.T).toarray()
        for r in range(sim.shape[0]):
            i = start + r
            row = sim[r]
            row[:i + 1] = 0.0
            j = np.nonzero(row >= floor)[0]
            if j.size:
                rows.append(np.full(j.size, i))
                cols.append(j)
                vals.append(row[j])

    if not rows:
        return pd.DataFrame(columns=['gvkey1', 'gvkey2', 'my_score'])

    i = np.concatenate(rows)
    j = np.concatenate(cols)
    v = np.concatenate(vals)
    return pd.DataFrame({'gvkey1': gvkeys[i], 'gvkey2': gvkeys[j],
                         'my_score': v})


# --------------------------------------------------------------- TNIC

def load_tnic(path):
    """Load TNIC and collapse both directions to (low gvkey, high gvkey)."""
    t = pd.read_csv(path, sep='\t')
    t = t[t.gvkey1 != t.gvkey2].dropna(subset=['score'])
    lo = np.minimum(t.gvkey1.values, t.gvkey2.values)
    hi = np.maximum(t.gvkey1.values, t.gvkey2.values)
    t = pd.DataFrame({'a': lo, 'b': hi, 'tnic_score': t.score.values})
    return t.drop_duplicates(['a', 'b'])


def join_scores(mine, tnic):
    m = mine.copy()
    m['a'] = np.minimum(m.gvkey1.values, m.gvkey2.values)
    m['b'] = np.maximum(m.gvkey1.values, m.gvkey2.values)
    m = m[['a', 'b', 'my_score']].drop_duplicates(['a', 'b'])
    return tnic.merge(m, on=['a', 'b'], how='inner')


def metrics(both, tnic_n):
    """
    Returns the numbers that matter.

    top1 is the headline: overall correlation is inflated by millions of
    unrelated pairs where both scores are ~0, which agree trivially. The
    top 1% of TNIC pairs is where industry membership is decided.
    """
    hi = both[both.tnic_score >= both.tnic_score.quantile(0.99)]
    return {
        'n_common': len(both),
        'overall_p': both.tnic_score.corr(both.my_score),
        'overall_s': both.tnic_score.corr(both.my_score, method='spearman'),
        'top1_p': hi.tnic_score.corr(hi.my_score),
        'top1_s': hi.tnic_score.corr(hi.my_score, method='spearman'),
        'my_mean': both.my_score.mean(),
        'tnic_mean': both.tnic_score.mean(),
        'my_p50': both.my_score.median(),
        'tnic_p50': both.tnic_score.median(),
        'missed_pct': (tnic_n - len(both)) / tnic_n,
    }


# --------------------------------------------------------------- sweep

def build(nouns_df, max_df, min_df):
    vec = CountVectorizer(binary=True, max_df=max_df, min_df=min_df)
    X = vec.fit_transform(nouns_df.nouns)
    vocab = X.shape[1]
    wpf = X.sum(axis=1).mean()
    X = normalize(X, norm='l2')
    return X, vocab, wpf


def cmd_sweep(nouns_path, tnic_path):
    d = pd.read_parquet(nouns_path)
    print(f"{len(d):,} firms with cached nouns")

    print("loading TNIC ...")
    tnic = load_tnic(tnic_path)
    print(f"  {len(tnic):,} unique TNIC pairs\n")

    grid = [(0.01,5),(0.02,5),(0.03,5),(0.05,5),
            (0.07,5),(0.10,5),(0.15,5),(0.20,5)]

    print(f"{'max_df':>7} {'min_df':>7} {'vocab':>7} {'w/firm':>7} "
          f"{'my_mean':>8} {'tnic_mu':>8} {'overall':>8} {'TOP1%':>8}")
    print('-' * 72)

    best = None
    for mx, mn in grid:
        X, vocab, wpf = build(d, mx, mn)
        mine = pairwise_above_floor(X, d.gvkey.values, quiet=True)
        both = join_scores(mine, tnic)
        r = metrics(both, len(tnic))
        print(f"{mx:>7.2f} {mn:>7d} {vocab:>7d} {wpf:>7.0f} "
              f"{r['my_mean']:>8.4f} {r['tnic_mean']:>8.4f} "
              f"{r['overall_p']:>8.3f} {r['top1_p']:>8.3f}")
        if best is None or r['top1_p'] > best[1]['top1_p']:
            best = ((mx, mn), r)

    (mx, mn), r = best
    print(f"\nbest: max_df={mx} min_df={mn}  top1%_pearson={r['top1_p']:.3f}")
    print("\nnote: watch my_mean vs tnic_mu. If yours is much higher, your")
    print("vectors share too much vocabulary and everything looks related.")


# --------------------------------------------------------------- single run

def cmd_one(nouns_path, tnic_path, out_path, max_df, min_df):
    d = pd.read_parquet(nouns_path)
    print(f"{len(d):,} firms")

    X, vocab, wpf = build(d, max_df, min_df)
    print(f"  {X.shape[0]:,} firms x {vocab:,} words, {wpf:.0f} words/firm")

    mine = pairwise_above_floor(X, d.gvkey.values)
    print(f"  {len(mine):,} pairs above floor {FLOOR}")
    mine.to_parquet(out_path, index=False)
    print(f"  wrote {out_path}")

    tnic = load_tnic(tnic_path)
    both = join_scores(mine, tnic)
    r = metrics(both, len(tnic))

    print(f"\n  pairs in common : {r['n_common']:,}")
    print(f"  overall pearson : {r['overall_p']:.4f}")
    print(f"  overall spearman: {r['overall_s']:.4f}")
    print(f"  TOP 1% pearson  : {r['top1_p']:.4f}   <-- the one that matters")
    print(f"  TOP 1% spearman : {r['top1_s']:.4f}")
    print(f"\n  my mean   {r['my_mean']:.4f}   median {r['my_p50']:.4f}")
    print(f"  TNIC mean {r['tnic_mean']:.4f}   median {r['tnic_p50']:.4f}")
    print(f"\n  TNIC pairs I have no score for: {r['missed_pct']:.1%}")


# --------------------------------------------------------------- entry

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1].lower()
    if cmd == 'tag':
        n_proc = int(sys.argv[4]) if len(sys.argv) > 4 else 1
        cmd_tag(sys.argv[2], sys.argv[3], n_proc)
    elif cmd == 'sweep':
        cmd_sweep(sys.argv[2], sys.argv[3])
    elif cmd == 'one':
        cmd_one(sys.argv[2], sys.argv[3], sys.argv[4],
                float(sys.argv[5]), int(sys.argv[6]))
    else:
        print(__doc__)
        sys.exit(1)