"""Mouse sample table for the MethAgingDB RRBS matrices: age in months, tissue, strain, condition, use flag."""
import pandas as pd, glob, re, numpy as np
M = pd.read_csv('./geo/mouse_geo_meta.csv', low_memory=False).set_index('GSM')
WK, DAY = 7 / 30.4375, 1 / 30.4375


def age_months(s):
    """Parse heterogeneous GEO age strings to months (postnatal; E18.5 -> slightly negative)."""
    if s is None or (isinstance(s, float) and np.isnan(s)): return np.nan
    s = str(s).strip().lower()
    if m := re.fullmatch(r'([\d.]+)\s*mo', s): return float(m.group(1))
    if m := re.fullmatch(r'month\s*([\d.]+)', s): return float(m.group(1))
    if m := re.fullmatch(r'([\d.]+)\s*(w|wk|weeks?|weeks old)( old)?', s): return float(m.group(1)) * WK
    if m := re.fullmatch(r'(?:wk|w)\s*([\d.]+)', s): return float(m.group(1)) * WK
    if m := re.fullmatch(r'([\d.]+) weeks?', s): return float(m.group(1)) * WK
    if m := re.fullmatch(r'p\s*([\d.]+)', s): return float(m.group(1)) * DAY
    if m := re.fullmatch(r'e\s*([\d.]+)', s): return (float(m.group(1)) - 19.5) * DAY
    if s == 'nine weeks': return 9 * WK
    return np.nan


rows = []
for f in sorted(glob.glob('./raw/x/*/*_count.csv')):
    tissue, gse = re.match(r'.*/([A-Za-z]+)_(GSE\d+)_count\.csv', f).groups()
    cols = open(f).readline().rstrip('\n').split(',')[1:]
    for c in cols:
        d = {'file': f, 'col': c, 'GSE': gse, 'tissue': tissue}
        if gse == 'GSE60012':
            # column label e.g. Liver_F_8w_NORMAL.1 / Liver_M_28w_old_castration_control.4
            lab = re.sub(r'\.\d+$', '', c)
            parts = lab.split('_') + ['nan', 'UNKNOWN']
            d.update(sex={'F': 'F', 'M': 'M'}.get(parts[1]), age_raw=parts[2], condition='_'.join(parts[3:-2]),
                     strain='C57BL/6')
        else:
            r = M.loc[c]
            d['sex'] = None
            if gse == 'GSE80672':
                d['age_raw'] = r.get('age (years)'); d['condition'] = 'CR' if r.get('diet') == 'Calorie Restricted' else 'NORMAL'
                d['sample_tissue'] = r.get('tissue')
            elif gse == 'GSE79775':
                t = str(r.get('title'))
                d['age_raw'] = 'wk9' if t.startswith('wk9') else r.get('developmental stage'); d['condition'] = 'NORMAL'
            elif gse == 'GSE52266':
                d['age_raw'] = r.get('age'); d['condition'] = r.get('source')
            elif gse == 'GSE89275':
                d['age_raw'] = None; d['condition'] = f"{r.get('genotype')}|{r.get('treatment')}"
            else:
                d['age_raw'] = r.get('age'); d['condition'] = 'NORMAL'
            d['strain'] = r.get('strain')
        rows.append(d)
S = pd.DataFrame(rows)
S['age_raw'] = S.age_raw.astype(str)
S['age_mo'] = [float(re.match(r'[\d.]+', a).group()) if g == 'GSE80672' and re.match(r'[\d.]+', a) else age_months(a)
               for g, a in zip(S.GSE, S.age_raw)]
# use flags
S['use'] = True
S.loc[S.GSE.isin(['GSE52266', 'GSE67507', 'GSE89275']), 'use'] = False          # no age variation / no age
S.loc[(S.GSE == 'GSE80672') & (S.tissue != 'Blood'), 'use'] = False              # cultured fibroblasts / iPSCs
S.loc[S.age_mo.isna(), 'use'] = False
S['control'] = S.condition.isin(['NORMAL'])
S['stage'] = pd.cut(S.age_mo, [-1, 1, 3, 12, 99], labels=['developmental (<1 mo)', 'juvenile (1-3 mo)', 'adult (3-12 mo)', 'old (>12 mo)'])
S.to_csv('./out/samples.csv', index=False)
u = S[S.use]
print(u.groupby(['GSE', 'tissue']).agg(n=('col', 'size'), ctrl=('control', 'sum'), amin=('age_mo', 'min'), amax=('age_mo', 'max')).round(2).to_string())
print('total usable', len(u), 'controls', u.control.sum())
print(S[~S.use].groupby(['GSE', 'tissue']).size())
print(S[S.age_mo.isna()][['GSE', 'col', 'age_raw']].head())
