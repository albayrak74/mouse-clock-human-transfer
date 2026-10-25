"""Sensitivity analyses: number of selected genes (k) and elastic-net penalty (alpha) for the leave-one-series-out
mouse analysis and for mouse->human transfer; brain-region breakdown of the human brain study."""
import sys, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
from m4_load import load_mouse, load_human, rel_age, inv_rel_age
from m5_models import fit_predict, TRANSFORMS
from scipy.stats import spearmanr
O = './out/'
Xm, Mm = load_mouse(); Xh, Mh = load_human()
genes = sorted(set(Xm.columns) & set(Xh.columns))
Xm, Xh = Xm.loc[:, ~Xm.columns.duplicated()][genes], Xh.loc[:, ~Xh.columns.duplicated()][genes]
c = Mm.control.values; Xm, Mm = Xm[c], Mm[c]; ym = rel_age(Mm.age_y, 'mouse')
keep = np.ones(len(Mh), bool)
for col, ok in [('diseasestatus', lambda s: s.str.contains('1|control|Control|Healthy', regex=True)),
                ('disease state', lambda s: s.str.contains('normal', case=False))]:
    if col in Mh:
        has = Mh[col].notna(); keep &= ~has.values | ok(Mh[col].fillna('')).values
keep &= Mh.age_y.notna().values
Xh, Mh = Xh[keep], Mh[keep]
R = []
for k, alpha in [(1000, 0.01), (500, 0.01), (2000, 0.01), (1000, 0.001), (1000, 0.1)]:
    pr = np.full(len(Mm), np.nan)
    for g in Mm.GSE.unique():
        te = Mm.GSE.values == g
        pr[te], _, _ = fit_predict(Xm[~te], ym[~te], Mm.dataset.values[~te], Xm[te], Mm.dataset.values[te], 'raw', k=k, alpha=alpha)
    R.append({'k': k, 'alpha': alpha, 'analysis': 'mouse LOGSE', 'target': 'all mice', 'transform': 'raw',
              'rho': spearmanr(Mm.age_mo, pr)[0]})
    for t in ['sample_z', 'rank', 'dataset_z', 'coral']:
        for g in sorted(Mh.GSE.unique()):
            i = Mh.GSE.values == g
            p, _, _ = fit_predict(Xm, ym, Mm.dataset.values, Xh[i], Mh.GSE.values[i], t, k=k, alpha=alpha)
            R.append({'k': k, 'alpha': alpha, 'analysis': 'mouse->human', 'target': g, 'transform': t,
                      'rho': spearmanr(Mh.age_y.values[i], p)[0] if np.std(p) > 0 else np.nan})
    print(k, alpha, flush=True)
pd.DataFrame(R).to_csv(O + 'S_sensitivity.csv', index=False)
# brain regions (GSE15745), from the main predictions
P = json.load(open(O + 'B_preds.json'))
m = Mh[Mh.GSE.values == 'GSE15745'].copy()
reg = m.tissue.str.lower().replace({'fctx': 'frontal cortex'}).values
Rg = []
for t in TRANSFORMS:
    p = np.array(P[f'{t}|GSE15745'])
    for r in sorted(set(reg)):
        j = reg == r
        if j.sum() >= 15:
            Rg.append({'transform': t, 'region': r, 'n': int(j.sum()),
                       'rho': spearmanr(m.age_y.values[j], p[j])[0] if np.std(p[j]) > 0 else np.nan,
                       'mean_score_resid': np.nan})
pd.DataFrame(Rg).to_csv(O + 'S_brain_regions.csv', index=False)
print('done')
