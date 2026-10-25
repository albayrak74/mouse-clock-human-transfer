"""Revision analyses after the reviewer round:
 (a) fraction of predictions at the bounds, per mouse scheme/unit and per human study/adaptation;
 (b) donor-level cluster bootstrap CIs for the brain study (GSE15745, 144 donors);
 (c) CORAL regularisation sensitivity (eps 0.1, 1, 10);
 (d) leave-one-series-out on a gene set common to all series;
 (e) human->human reference with all five feature versions (slow; last)."""
import sys, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
import m5_models as M5
from m4_load import load_mouse, load_human, rel_age, inv_rel_age
from m5_models import fit_predict, TRANSFORMS
from scipy.stats import spearmanr
O = './out/'
rng = np.random.default_rng(20261024)

# ---------- (a) saturation
P = pd.read_csv(O + 'A_mouse_preds.csv')
cols = {s: f'pred_{s}' for s in ['random5fold', 'LODO', 'LOGSE', 'LOTO']}
hi = max(P[c].max() for c in cols.values()); lo = min(P[c].min() for c in cols.values())
rows = []
for s, c in cols.items():
    at = (P[c] >= hi - 1e-6) | (P[c] <= lo + 1e-6)
    rows.append({'split': s, 'dataset': 'ALL', 'pct_bound': 100 * at.mean()})
    for d, g in P.groupby('dataset'):
        a = (g[c] >= hi - 1e-6) | (g[c] <= lo + 1e-6); rows.append({'split': s, 'dataset': d, 'pct_bound': 100 * a.mean()})
pd.DataFrame(rows).to_csv(O + 'A_saturation.csv', index=False)
Pj = json.load(open(O + 'B_preds.json'))
hlo = min(min(v) for v in Pj.values()); hhi = max(max(v) for v in Pj.values())
rows = []
for k, v in Pj.items():
    t, g = k.split('|'); v = np.array(v)
    rows.append({'transform': t, 'GSE': g, 'pct_bound': 100 * ((v <= hlo + 1e-9) | (v >= hhi - 1e-9)).mean(), 'n_distinct': len(np.unique(v))})
pd.DataFrame(rows).to_csv(O + 'B_saturation.csv', index=False)
print('saturation done', flush=True)

# ---------- (b) donor bootstrap for GSE15745
m = pd.read_csv(O + 'hmeta_GSE15745.csv', low_memory=False)
keep = m.age_y.notna().values
donor = m['!Sample_title'].str.extract(r'(\d+)')[0].values[keep]; age = m.age_y.values[keep]
ud = np.unique(donor); idx = {d: np.where(donor == d)[0] for d in ud}
rows = []
for t in TRANSFORMS:
    p = np.array(Pj[f'{t}|GSE15745']); assert len(p) == len(age)
    bs = []
    for b in range(1000):
        s = np.concatenate([idx[d] for d in rng.choice(ud, len(ud), replace=True)])
        if np.std(p[s]) > 0: bs.append(spearmanr(age[s], p[s])[0])
    rows.append({'transform': t, 'GSE': 'GSE15745', 'n_donors': len(ud), 'rho': spearmanr(age, p)[0] if np.std(p) > 0 else np.nan,
                 'rho_lo': np.percentile(bs, 2.5) if bs else np.nan, 'rho_hi': np.percentile(bs, 97.5) if bs else np.nan})
pd.DataFrame(rows).to_csv(O + 'B_rho_ci_donor.csv', index=False)
print('donor bootstrap done', flush=True)

# ---------- shared loading for model reruns
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
Xh, Mh = Xh[keep], Mh[keep]; yh = rel_age(Mh.age_y, 'human')

# ---------- (c) CORAL eps sensitivity
rows = []
_coral = M5._coral
for eps in [0.1, 1.0, 10.0]:
    M5._coral = lambda Xs, Xt, eps=eps: _coral(Xs, Xt, eps=eps)
    for g in sorted(Mh.GSE.unique()):
        i = Mh.GSE.values == g
        p, _, _ = fit_predict(Xm, ym, Mm.dataset.values, Xh[i], Mh.GSE.values[i], 'coral')
        rows.append({'eps': eps, 'GSE': g, 'rho': spearmanr(Mh.age_y.values[i], p)[0] if np.std(p) > 0 else np.nan,
                     'mae_years': np.mean(np.abs(inv_rel_age(p, 'human') - Mh.age_y.values[i]))})
    print('coral eps', eps, flush=True)
M5._coral = _coral
pd.DataFrame(rows).to_csv(O + 'S_coral_eps.csv', index=False)

# ---------- (d) leave-one-series-out on a common gene set
common = np.ones(Xm.shape[1], bool)
for u in Mm.dataset.unique():
    common &= (Xm[Mm.dataset.values == u].notna().mean() >= 0.7).values
Xc = Xm.loc[:, common]; print('common genes', int(common.sum()), flush=True)
pr = np.full(len(Mm), np.nan)
for g in Mm.GSE.unique():
    te = Mm.GSE.values == g
    pr[te], _, _ = fit_predict(Xc[~te], ym[~te], Mm.dataset.values[~te], Xc[te], Mm.dataset.values[te], 'raw')
rows = [{'dataset': 'ALL', 'n_genes': int(common.sum()), 'rho': spearmanr(Mm.age_mo, pr)[0],
         'medae': np.median(np.abs(inv_rel_age(pr, 'mouse') * 12 - Mm.age_mo.values))}]
for d in np.unique(Mm.dataset):
    i = Mm.dataset.values == d
    rows.append({'dataset': d, 'n_genes': int(common.sum()), 'rho': spearmanr(Mm.age_mo.values[i], pr[i])[0] if np.std(pr[i]) > 0 else np.nan,
                 'medae': np.median(np.abs(inv_rel_age(pr[i], 'mouse') * 12 - Mm.age_mo.values[i]))})
pd.DataFrame(rows).to_csv(O + 'S_logse_common.csv', index=False)
print('common-gene LOGSE done', flush=True)

# ---------- (e) human->human with all five versions
E = []
for t in TRANSFORMS:
    if t in ('raw', 'dataset_z'): continue
    for gtr in sorted(Mh.GSE.unique()):
        tr = Mh.GSE.values == gtr
        if tr.sum() < 100: continue
        for gte in sorted(Mh.GSE.unique()):
            if gte == gtr: continue
            te = Mh.GSE.values == gte
            p, _, _ = fit_predict(Xh[tr], yh[tr], Mh.GSE.values[tr], Xh[te], Mh.GSE.values[te], t)
            E.append({'transform': t, 'train': gtr, 'test': gte, 'rho': spearmanr(yh[te], p)[0] if np.std(p) > 0 else np.nan,
                      'mae_years': np.mean(np.abs(inv_rel_age(p, 'human') - Mh.age_y.values[te]))})
    print('E', t, flush=True)
E0 = pd.read_csv(O + 'E_human_reference_v2.csv')[['transform', 'train', 'test', 'rho', 'mae_years']]
pd.concat([E0, pd.DataFrame(E)]).to_csv(O + 'E_human_reference_v3.csv', index=False)
print('done', flush=True)
