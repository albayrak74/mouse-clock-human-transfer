"""Main analyses for Makale-9.
A  within-mouse transportability: random 5-fold CV vs leave-one-dataset-out (LODO) vs leave-one-tissue-out (LOTO)
B  mouse -> human transfer, per human dataset, for five feature transforms (domain adaptation)
C  failure modes: by human tissue match, platform, age stratum
D  few-shot recalibration with k labelled human samples
E  human -> human reference (train on one human set, test on others) as a ceiling"""
import sys, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
from m4_load import load_mouse, load_human, rel_age, inv_rel_age
from m5_models import fit_predict, metrics, TRANSFORMS
from sklearn.model_selection import KFold
O = './out/'
rng = np.random.default_rng(20261024)
Xm, Mm = load_mouse(); Xh, Mh = load_human()
genes = sorted(set(Xm.columns) & set(Xh.columns))
Xm, Xh = Xm.loc[:, ~Xm.columns.duplicated()][genes], Xh.loc[:, ~Xh.columns.duplicated()][genes]
ctrl = Mm.control.values
Xm, Mm = Xm[ctrl], Mm[ctrl]
ym = rel_age(Mm.age_y, 'mouse')
# human sample filters: controls / healthy only where disease status is recorded
keep = np.ones(len(Mh), bool)
for col, ok in [('diseasestatus', lambda s: s.str.contains('1|control|Control|Healthy', regex=True)),
                ('disease state', lambda s: s.str.contains('normal', case=False))]:
    if col in Mh:
        has = Mh[col].notna(); keep &= ~has.values | ok(Mh[col].fillna('')).values
keep &= Mh.age_y.notna().values
Xh, Mh = Xh[keep], Mh[keep]
yh = rel_age(Mh.age_y, 'human')
print('mouse controls', len(Mm), 'datasets', Mm.dataset.nunique(), '| human', len(Mh), Mh.GSE.value_counts().to_dict(), '| shared genes', len(genes), flush=True)
R = {}

# ---------- A: within mouse
def mouse_eval(split_name, groups):
    rows, preds = [], np.full(len(Mm), np.nan)
    for g in np.unique(groups):
        te = groups == g; tr = ~te
        p, _, _ = fit_predict(Xm[tr], ym[tr], Mm.dataset.values[tr], Xm[te], Mm.dataset.values[te], 'raw')
        preds[te] = p
    for d in np.unique(Mm.dataset):
        i = Mm.dataset.values == d
        mm = metrics(Mm.age_mo.values[i], inv_rel_age(preds[i], 'mouse') * 12); mm.update(split=split_name, dataset=d); rows.append(mm)
    allm = metrics(Mm.age_mo.values, inv_rel_age(preds, 'mouse') * 12); allm.update(split=split_name, dataset='ALL'); rows.append(allm)
    return rows, preds

kf = np.zeros(len(Mm), int)
for f, (_, te) in enumerate(KFold(5, shuffle=True, random_state=1).split(Mm)): kf[te] = f
import os
A = []
if os.path.exists(O + 'A_mouse_transport.csv'):
    print('A cached', flush=True)
else:
  for name, grp in [('random5fold', kf), ('LODO', Mm.dataset.values), ('LOGSE', Mm.GSE.values), ('LOTO', Mm.tissue.values)]:
      rows, p = mouse_eval(name, grp); A += rows; Mm[f'pred_{name}'] = inv_rel_age(p, 'mouse') * 12
      print(name, {k: round(v, 3) for k, v in rows[-1].items() if isinstance(v, float)}, flush=True)
  pd.DataFrame(A).to_csv(O + 'A_mouse_transport.csv', index=False)
  Mm.to_csv(O + 'A_mouse_preds.csv')

# ---------- B/C: mouse -> human, five transforms
B, P = [], {}
for t in TRANSFORMS:
    for g in sorted(Mh.GSE.unique()):
        i = Mh.GSE.values == g
        p, used, _ = fit_predict(Xm, ym, Mm.dataset.values, Xh[i], Mh.GSE.values[i], t)
        P[(t, g)] = p
        mm = metrics(yh[i], p); mm.update(transform=t, GSE=g, platform=Mh.platform.values[i][0], tissue=Mh.tissue_group.values[i][0],
                                         mae_years=np.mean(np.abs(inv_rel_age(p, 'human') - Mh.age_y.values[i])), n_genes=len(used))
        B.append(mm)
    print(t, pd.DataFrame(B)[lambda d: d['transform'] == t][['GSE', 'rho', 'mae_years']].round(3).values.tolist(), flush=True)
pd.DataFrame(B).to_csv(O + 'B_mouse_to_human.csv', index=False)
pdict = {f'{t}|{g}': list(map(float, v)) for (t, g), v in P.items()}
json.dump(pdict, open(O + 'B_preds.json', 'w'))

# age strata (failure by age range)
C = []
for (t, g), p in P.items():
    i = Mh.GSE.values == g; a = Mh.age_y.values[i]
    for lab, lo, hi in [('<30', 0, 30), ('30-60', 30, 60), ('>=60', 60, 200)]:
        j = (a >= lo) & (a < hi)
        if j.sum() >= 15:
            mm = metrics(a[j], inv_rel_age(p[j], 'human')); mm.update(transform=t, GSE=g, stratum=lab); C.append(mm)
pd.DataFrame(C).to_csv(O + 'C_age_strata.csv', index=False)

# ---------- D: few-shot recalibration (linear map on k labelled target samples)
D = []
for (t, g), p in P.items():
    if t not in ('raw', 'dataset_z'): continue
    i = Mh.GSE.values == g; y = yh[i]; n = i.sum()
    for k in [5, 10, 20, 50]:
        if k >= n - 10: continue
        res = []
        for b in range(200):
            s = rng.choice(n, k, replace=False); rest = np.setdiff1d(np.arange(n), s)
            A_, c_ = np.polyfit(p[s], y[s], 1) if np.std(p[s]) > 0 else (0, y[s].mean())
            q = A_ * p[rest] + c_
            res.append(np.mean(np.abs(inv_rel_age(q, 'human') - inv_rel_age(y[rest], 'human'))))
        D.append({'transform': t, 'GSE': g, 'k': k, 'mae_years_mean': np.mean(res), 'mae_years_lo': np.percentile(res, 2.5), 'mae_years_hi': np.percentile(res, 97.5)})
pd.DataFrame(D).to_csv(O + 'D_fewshot.csv', index=False)

# ---------- E: human -> human reference (same gene space, raw)
E = []
for gtr in sorted(Mh.GSE.unique()):
    tr = Mh.GSE.values == gtr
    if tr.sum() < 100: continue
    for gte in sorted(Mh.GSE.unique()):
        if gte == gtr: continue
        te = Mh.GSE.values == gte
        p, _, _ = fit_predict(Xh[tr], yh[tr], Mh.GSE.values[tr], Xh[te], Mh.GSE.values[te], 'raw')
        mm = metrics(yh[te], p); mm.update(train=gtr, test=gte, mae_years=np.mean(np.abs(inv_rel_age(p, 'human') - Mh.age_y.values[te]))); E.append(mm)
pd.DataFrame(E).to_csv(O + 'E_human_reference.csv', index=False)
print('done')
