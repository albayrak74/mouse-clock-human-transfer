"""Follow-up to m6: few-shot with clipped back-transform and a 'k-sample mean' baseline; age strata for adapted
transforms; bootstrap CIs for mouse->human rho; human->human reference with dataset_z as well as raw."""
import sys, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
from m4_load import load_human, rel_age, inv_rel_age
from m5_models import fit_predict, metrics
from scipy.stats import spearmanr
O = './out/'
rng = np.random.default_rng(20261024)
Xh, Mh = load_human()
keep = np.ones(len(Mh), bool)
for col, ok in [('diseasestatus', lambda s: s.str.contains('1|control|Control|Healthy', regex=True)),
                ('disease state', lambda s: s.str.contains('normal', case=False))]:
    if col in Mh:
        has = Mh[col].notna(); keep &= ~has.values | ok(Mh[col].fillna('')).values
keep &= Mh.age_y.notna().values
Xh, Mh = Xh[keep], Mh[keep]
yh = rel_age(Mh.age_y, 'human')
LO, HI = rel_age(0, 'human'), 0.0
P = {tuple(k.split('|')): np.array(v) for k, v in json.load(open(O + 'B_preds.json')).items()}

# bootstrap CI for rho
CI = []
for (t, g), p in P.items():
    a = Mh.age_y.values[Mh.GSE.values == g]; n = len(a); bs = []
    for b in range(1000):
        s = rng.integers(0, n, n)
        if np.std(p[s]) > 0: bs.append(spearmanr(a[s], p[s])[0])
    CI.append({'transform': t, 'GSE': g, 'rho': spearmanr(a, p)[0] if np.std(p) > 0 else np.nan,
               'rho_lo': np.percentile(bs, 2.5) if bs else np.nan, 'rho_hi': np.percentile(bs, 97.5) if bs else np.nan,
               'frac_constant': np.mean(p == p[0])})
pd.DataFrame(CI).to_csv(O + 'B_rho_ci.csv', index=False)

# few-shot: linear map in log relative age, clipped to [birth, max lifespan]; baseline = mean of the k labelled ages
D = []
for (t, g), p in P.items():
    i = Mh.GSE.values == g; y = yh[i]; a = Mh.age_y.values[i]; n = i.sum()
    for k in [5, 10, 20, 50]:
        if k >= n - 10: continue
        res, base = [], []
        for b in range(500):
            s = rng.choice(n, k, replace=False); rest = np.setdiff1d(np.arange(n), s)
            if np.std(p[s]) > 0:
                A_, c_ = np.polyfit(p[s], y[s], 1); q = np.clip(A_ * p[rest] + c_, LO, HI)
            else:
                q = np.full(len(rest), y[s].mean())
            res.append(np.mean(np.abs(inv_rel_age(q, 'human') - a[rest])))
            base.append(np.mean(np.abs(a[s].mean() - a[rest])))
        res, base = np.array(res), np.array(base)
        D.append({'transform': t, 'GSE': g, 'k': k, 'mae': np.median(res), 'mae_lo': np.percentile(res, 2.5), 'mae_hi': np.percentile(res, 97.5),
                  'mae_baseline': np.median(base), 'p_better': np.mean(res < base)})
pd.DataFrame(D).to_csv(O + 'D_fewshot_v2.csv', index=False)

# age strata on all transforms (rank correlation within stratum)
C = []
for (t, g), p in P.items():
    a = Mh.age_y.values[Mh.GSE.values == g]
    for lab, lo, hi in [('<40', 0, 40), ('40-60', 40, 60), ('>=60', 60, 200)]:
        j = (a >= lo) & (a < hi)
        if j.sum() >= 15:
            C.append({'transform': t, 'GSE': g, 'stratum': lab, 'n': int(j.sum()),
                      'rho': spearmanr(a[j], p[j])[0] if np.std(p[j]) > 0 else np.nan})
pd.DataFrame(C).to_csv(O + 'C_age_strata_v2.csv', index=False)

# human -> human, raw and dataset_z
E = []
for t in ['raw', 'dataset_z']:
    for gtr in sorted(Mh.GSE.unique()):
        tr = Mh.GSE.values == gtr
        if tr.sum() < 100: continue
        for gte in sorted(Mh.GSE.unique()):
            if gte == gtr: continue
            te = Mh.GSE.values == gte
            p, _, _ = fit_predict(Xh[tr], yh[tr], Mh.GSE.values[tr], Xh[te], Mh.GSE.values[te], t)
            mm = metrics(yh[te], p); mm.update(transform=t, train=gtr, test=gte, mae_years=np.mean(np.abs(inv_rel_age(p, 'human') - Mh.age_y.values[te])))
            E.append(mm)
pd.DataFrame(E).to_csv(O + 'E_human_reference_v2.csv', index=False)
print('done')
