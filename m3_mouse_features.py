"""Gene-level promoter methylation (mean beta of CpGs in TSS -1500/+500) for every usable mouse RRBS sample.
Streams each matrix in chunks; outputs genes x samples mean and CpG-count matrices (mouse symbols)."""
import pandas as pd, numpy as np, sys, os
O = './out/'
S = pd.read_csv(O + 'samples.csv')
S = S[S.use]
P = pd.read_csv('./ann/mm10_promoters.csv')
genes = P.gene.values; G = len(genes)
W = {c: (d.lo.values, d.hi.values, d.index.values) for c, d in P.reset_index(drop=True).groupby('chr')}
W = {c: (lo[np.argsort(lo)], hi[np.argsort(lo)], ix[np.argsort(lo)]) for c, (lo, hi, ix) in W.items()}


def map_rows(chrs, pos):
    """Return (row_index, gene_index) pairs for CpGs that fall in promoter windows."""
    R, Gi = [], []
    for c in np.unique(chrs):
        if c not in W: continue
        lo, hi, ix = W[c]
        m = np.where(chrs == c)[0]; p = pos[m]
        j = np.searchsorted(lo, p, side='right') - 1
        for k in range(4):                               # windows are 2 kb; at most a few can overlap
            jj = j - k; ok = jj >= 0
            jj2 = np.where(ok, jj, 0)
            hit = ok & (lo[jj2] <= p) & (p <= hi[jj2])
            R.append(m[hit]); Gi.append(ix[jj2[hit]])
    return (np.concatenate(R), np.concatenate(Gi)) if R else (np.array([], int), np.array([], int))


for f, d in S.groupby('file'):
    tag = os.path.basename(f).replace('_count.csv', '')
    if os.path.exists(O + f'feat_{tag}.npz'): continue
    cols = d.col.tolist()
    hdr = open(f).readline().rstrip('\n').split(',')
    use = [0] + [hdr.index(c) for c in cols]
    sm = np.zeros((G, len(cols))); n = np.zeros((G, len(cols)))
    for ch in pd.read_csv(f, usecols=use, chunksize=200000, index_col=0):
        ch = ch[cols]
        idx_s = ch.index.astype(str)
        if idx_s[0].startswith('gi|'):   # GSE80672 uses RefSeq GRCm38 accessions: NC_000067.6 = chr1 ... NC_000086.7 = chrX
            acc = idx_s.str.extract(r'NC_0000(\d\d)\.\d+\|:(\d+)')
            num = pd.to_numeric(acc[0], errors='coerce') - 66
            name = np.where(num <= 19, 'chr' + num.astype('Int64').astype(str), np.where(num == 20, 'chrX', np.where(num == 21, 'chrY', 'NA')))
            idx_s = pd.Index(pd.Series(name, index=None).astype(str) + '_' + acc[1].astype(str))
        ix = idx_s.str.rsplit('_', n=1, expand=True)
        chrs = np.asarray(ix.get_level_values(0)); pos = pd.to_numeric(np.asarray(ix.get_level_values(1)), errors='coerce')
        keep = ~np.isnan(pos) & pd.Series(chrs).astype(str).str.match(r'^chr[0-9XY]+$').fillna(False).values.astype(bool)
        ch = ch[keep]; chrs = chrs[keep]; pos = pos[keep].astype(np.int64)
        r, g = map_rows(chrs, pos)
        if len(r) == 0: continue
        v = ch.values[r].astype(float)
        ok = ~np.isnan(v)
        np.add.at(sm, g, np.where(ok, v, 0.0)); np.add.at(n, g, ok)
    np.savez_compressed(O + f'feat_{tag}.npz', mean=np.where(n > 0, sm / np.maximum(n, 1), np.nan).astype(np.float32),
                        n=n.astype(np.int32), cols=np.array(cols), genes=genes)
    cov = (n > 0).mean(0)
    print(tag, len(cols), 'genes covered per sample median', int(np.median((n > 0).sum(0))), flush=True)
