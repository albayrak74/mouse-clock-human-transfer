"""Human array data -> gene-level promoter methylation (human symbols) + sample table.
450K probes: hg19 MAPINFO inside TSS -1500/+500 windows (same rule as mouse). 27K probes: annotated gene symbol
(all 27K probes are promoter-proximal; the hg18 coordinates are not lifted)."""
import gzip, re, numpy as np, pandas as pd, os
H = './human/'; O = './out/'
P = pd.read_csv('./ann/hg19_promoters.csv')
genes = P.gene.values; gidx = {g: i for i, g in enumerate(genes)}


def gpl_table(path):
    lines = open(path, errors='ignore').read().split('\n')
    i = next(k for k, l in enumerate(lines) if l.startswith('ID\t'))
    from io import StringIO
    return pd.read_csv(StringIO('\n'.join(l for l in lines[i:] if l and not l.startswith('!'))), sep='\t', low_memory=False)


def probe_map_450k():
    t = gpl_table(H + 'GPL13534.txt')[['ID', 'CHR', 'MAPINFO']].dropna()
    t['chr'] = 'chr' + t.CHR.astype(str); t['pos'] = t.MAPINFO.astype(int)
    pairs = []
    for c, d in P.groupby('chr'):
        q = t[t.chr == c]
        if q.empty: continue
        lo, hi = d.lo.values, d.hi.values; o = np.argsort(lo); lo, hi, gi = lo[o], hi[o], d.gene.values[o]
        j = np.searchsorted(lo, q.pos.values, side='right') - 1
        for k in range(4):
            jj = j - k; ok = jj >= 0; jj2 = np.where(ok, jj, 0)
            hit = ok & (lo[jj2] <= q.pos.values) & (q.pos.values <= hi[jj2])
            pairs += list(zip(q.ID.values[hit], gi[jj2[hit]]))
    return pd.DataFrame(pairs, columns=['ID', 'gene'])


def probe_map_27k():
    t = gpl_table(H + 'GPL8490.txt')[['ID', 'Symbol']].dropna()
    t = t[t.Symbol.isin(gidx)]
    return t.rename(columns={'Symbol': 'gene'})


def read_series(f, keep=None):
    meta, ids, rows, cols = {}, None, [], None
    with gzip.open(f, 'rt', errors='ignore') as fh:
        for line in fh:
            if line.startswith('!Sample_geo_accession'): ids = [x.strip('"') for x in line.rstrip('\n').split('\t')[1:]]
            elif line.startswith('!Sample_characteristics_ch1') or line.startswith('!Sample_source_name_ch1') or line.startswith('!Sample_title'):
                key = line.split('\t')[0]
                for i, v in enumerate(x.strip('"') for x in line.rstrip('\n').split('\t')[1:]):
                    d = meta.setdefault(i, {})
                    if key == '!Sample_characteristics_ch1' and ':' in v:
                        k, val = v.split(':', 1); d[k.strip().lower()] = val.strip()
                    else: d[key] = v
            elif line.startswith('"ID_REF"'):
                cols = [x.strip('"') for x in line.rstrip('\n').split('\t')[1:]]; break
        parts = []
        for ch in pd.read_csv(fh, sep='\t', index_col=0, comment='!', header=None, chunksize=50000, quotechar='"'):
            ch.index = ch.index.astype(str).str.strip('"')
            if keep is not None: ch = ch[ch.index.isin(keep)]
            parts.append(ch.apply(pd.to_numeric, errors='coerce').astype(np.float32))
        X = pd.concat(parts)
    X.columns = cols
    m = pd.DataFrame([meta[i] for i in range(len(ids))]); m['GSM'] = ids
    return m, X


SETS = {'GSE20242': ('27K', 'Monocytes (CD14+)'), 'GSE41037': ('27K', 'Blood'), 'GSE15745': ('27K', 'Brain'),
        'GSE40279': ('450K', 'Blood'), 'GSE61258': ('450K', 'Liver'), 'GSE50498': ('450K', 'Muscle'), 'GSE52401': ('450K', 'Lung')}
maps = {'27K': probe_map_27k(), '450K': probe_map_450k()}
for k, v in maps.items(): print(k, 'probes mapped', v.ID.nunique(), 'genes', v.gene.nunique())
allmeta = []
for g, (plat, tis) in SETS.items():
    if os.path.exists(O + f'hfeat_{g}.npz'): continue
    f = [x for x in os.listdir(H) if x.startswith(g) and x.endswith('series_matrix.txt.gz')][0]
    m, X = read_series(H + f, keep=set(maps[plat].ID))
    pm = maps[plat]; pm = pm[pm.ID.isin(X.index)]
    Xg = X.loc[pm.ID.values]; Xg.index = pm.gene.values
    Gm = Xg.groupby(level=0).mean()
    n = Xg.notna().groupby(level=0).sum()
    M = np.full((len(genes), X.shape[1]), np.nan, np.float32); N = np.zeros_like(M, dtype=np.int32)
    ii = [gidx[x] for x in Gm.index]
    M[ii] = Gm.values; N[ii] = n.values
    m['GSE'] = g; m['platform'] = plat; m['tissue_group'] = tis
    agecol = next((c for c in m.columns if c.startswith('age')), None)
    m['age_y'] = pd.to_numeric(m[agecol], errors='coerce') if agecol else np.nan
    for alt in ['age (y)', 'age']:   # GSE15745 records age under two keys depending on brain region
        if alt in m: m['age_y'] = m['age_y'].fillna(pd.to_numeric(m[alt], errors='coerce'))
    np.savez_compressed(O + f'hfeat_{g}.npz', mean=M, n=N, cols=np.array(X.columns), genes=genes)
    m.to_csv(O + f'hmeta_{g}.csv', index=False)
    print(g, plat, tis, X.shape, 'genes', len(ii), 'age col', agecol, 'age range', m.age_y.min(), m.age_y.max(), flush=True)
