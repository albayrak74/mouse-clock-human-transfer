"""All numbers quoted in the manuscript, computed from the result files -> out/V.json."""
import json, sys, os, numpy as np, pandas as pd
sys.path.insert(0, '.')
O = './out/'
V = {}
f2 = lambda x: f'{x:.2f}'; f1 = lambda x: f'{x:.1f}'
S = pd.read_csv(O + 'samples.csv'); S = S[S.use]
C_ = S[S.control]
V['n_mouse_all'] = f'{len(S):,}'; V['n_mouse'] = f'{len(C_):,}'
V['n_units'] = str(C_.groupby(['tissue', 'GSE']).ngroups); V['n_gse_mouse'] = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven'][C_.GSE.nunique()]
V['n_tissue'] = str(C_.tissue.nunique()); V['age_max_mo'] = f'{C_.age_mo.max():.0f}'
A = pd.read_csv(O + 'A_mouse_transport.csv')
for sp in ['random5fold', 'LODO', 'LOGSE', 'LOTO']:
    r = A[(A.split == sp) & (A.dataset == 'ALL')].iloc[0]; k = 'random' if sp == 'random5fold' else sp
    V[f'A_rho_{k}'] = f2(r.rho); V[f'A_medae_{k}'] = f1(r.medae); V[f'A_bias_{k}'] = f1(r.bias)
g = lambda d, s: f2(A[(A.dataset == d) & (A.split == s)].rho.iloc[0])
V['A_liv_thompson'] = g('Liver_GSE120137', 'LODO'); V['A_liv_gse60012'] = g('Liver_GSE60012', 'LODO'); V['A_liv_gse58129'] = g('Liver_GSE58129', 'LODO')
V['A_liv_stubbs'] = g('Liver_GSE93957', 'LODO'); V['A_liv_cannon'] = g('Liver_GSE79775', 'LODO')
V['A_cer_logse'] = g('Cerebellum_GSE60012', 'LOGSE'); V['A_spl_logse'] = g('Spleen_GSE60012', 'LOGSE')
# check the claim "blood, lung and cortex >0.8 under every scheme"
for d in ['Blood_GSE120137', 'Blood_GSE80672', 'Cortex_GSE93957']:
    assert A[A.dataset == d].rho.min() > 0.7, d
assert A[A.dataset.str.startswith('Lung')].rho.min() > 0.6
# human
from m4_load import load_human
from helpers import load_ci, load_E, sat_pct, best_adaptation
CI = load_ci(); B = pd.read_csv(O + 'B_mouse_to_human.csv')
V['n_human'] = f"{int(B[B['transform'] == 'raw'].n.sum()):,}"
NH = B[B['transform'] == 'raw'].set_index('GSE').n
V['n_const_raw'] = ['zero', 'one', 'two', 'three', 'four', 'five', 'six'][int(CI[(CI['transform'] == 'raw') & CI.rho.isna()].shape[0])]
raw27 = CI[(CI['transform'] == 'raw') & CI.GSE.isin(['GSE15745', 'GSE20242', 'GSE41037'])].rho
V['B_raw_27k_min'], V['B_raw_27k_max'] = f2(raw27.min()), f2(raw27.max())
ad = CI[CI['transform'] != 'raw'].copy(); ad = ad[[sat_pct(g, t) <= 50 for g, t in zip(ad.GSE, ad['transform'])]]
best = ad.loc[ad.groupby('GSE').rho.idxmax()].set_index('GSE')
V['B_best_min'], V['B_best_max'] = f2(best.rho.min()), f2(best.rho.max())
TAG = {'GSE50498': 'mus', 'GSE61258': 'liv', 'GSE41037': 'b27', 'GSE15745': 'brn', 'GSE20242': 'mon', 'GSE40279': 'b450'}
for gse, t in TAG.items():
    r = best.loc[gse]; V[f'B_{t}_best'], V[f'B_{t}_lo'], V[f'B_{t}_hi'] = f2(r.rho), f2(r.rho_lo), f2(r.rho_hi); V[f'B_{t}_tr'] = r['transform']
V['B_b27_dz'] = f2(CI[(CI.GSE == 'GSE41037') & (CI['transform'] == 'dataset_z')].rho.iloc[0])
dz = B[B['transform'] == 'dataset_z'].mae_years
V['B_mae_dz_min'], V['B_mae_dz_max'] = f'{dz.min():.0f}', f'{dz.max():.0f}'
V["B_dz_max"] = f2(CI[CI["transform"] == "dataset_z"].rho.max())
V["B_dz_o_min"] = f2(CI[(CI["transform"] == "dataset_z") & (CI.GSE != "GSE41037")].rho.min())
E = load_E()
V['E_rho_min'], V['E_rho_max'] = f2(E.rho.min()), f2(E.rho.max())
V['E_b450_2nd'] = f2(sorted(E[E.test == 'GSE40279'].rho)[1])
C_ = pd.read_csv(O + 'C_age_strata_v2.csv'); _m = C_.loc[C_.rho.idxmax()]
V['C_all_max'], V['C_all_max_n'] = f2(_m.rho), str(int(_m.n))
Er = E[E['transform'] == 'raw']; V['E_mae_min'], V['E_mae_max'] = f'{Er.mae_years.min():.0f}', f'{Er.mae_years.max():.0f}'
for gse, t in TAG.items():
    e = E[E.test == gse].rho; V[f'E_{t}_min'], V[f'E_{t}_max'] = f2(e.min()), f2(e.max())
D = pd.read_csv(O + 'D_fewshot_v2.csv'); d20 = D[D.k == 20]
for gse, t in TAG.items():
    r = d20[(d20.GSE == gse) & (d20['transform'] == best.loc[gse, 'transform'])].iloc[0]
    V[f'D_{t}_base'], V[f'D_{t}_best'], V[f'D_{t}_p'] = f1(d20[d20.GSE == gse].mae_baseline.mean()), f1(r.mae), f'{100 * r.p_better:.0f}'
    r5 = D[(D.k == 5) & (D.GSE == gse) & (D['transform'] == best.loc[gse, 'transform'])].iloc[0]
    V[f'D5_{t}_hi'], V[f'D5_{t}_base'] = f1(r5.mae_hi), f1(r5.mae_baseline)
C = pd.read_csv(O + 'C_age_strata_v2.csv'); cz = C[C['transform'] == 'dataset_z']
V['C_dz_min'], V['C_dz_max'] = f2(cz.rho.min()), f2(cz.rho.max()); V['C_n_tot'] = str(len(cz)); V['C_n_pos'] = str(int((cz.rho > 0.3).sum()))
X, M = load_human()
ages = pd.concat([pd.read_csv(O + f'hmeta_{g}.csv', low_memory=False).age_y for g in NH.index])
from helpers import human_ages
ha = pd.concat([human_ages(g) for g in NH.index]); V['h_age_min'], V['h_age_max'] = f'{ha.min():.0f}', f'{ha.max():.0f}'
import glob
z = np.load(O + 'feat_Liver_GSE120137.npz', allow_pickle=True); V['n_ortho'] = f"{len(pd.read_csv('./ann/ortho_1to1.csv')):,}"
from m4_load import load_mouse
Xm, Mm = load_mouse(); Xm = Xm[Mm.control.values]
V['mouse_genes_med'] = f'{int(Xm.notna().sum(axis=1).median()):,}'
json.dump(V, open(O + 'V.json', 'w'), indent=1, ensure_ascii=False)
for k, v in V.items(): print(k, v)
print({g: int(n) for g, n in NH.items()})
# unit descriptors (who transports poorly)
from scipy.stats import spearmanr as _sp
G = pd.read_csv(O + 'A_unit_descriptors.csv', index_col=0)
for sp in ['LODO', 'LOGSE']:
    v = A[(A.split == sp) & (A.dataset != 'ALL')].set_index('dataset').rho.reindex(G.index)
    V[f'U_{sp}_span'] = f2(_sp(v, G.span)[0]); V[f'U_{sp}_cov'] = f2(_sp(v, G['cov'])[0]); V[f'U_{sp}_n'] = f2(_sp(v, G.n)[0])
V['U_cov_min'], V['U_cov_max'] = f'{int(G["cov"].min()):,}', f'{int(G["cov"].max()):,}'
json.dump(V, open(O + 'V.json', 'w'), indent=1, ensure_ascii=False)
# sensitivity
S = pd.read_csv(O + 'S_sensitivity.csv')
lg = S[S.analysis == 'mouse LOGSE'].rho
V['S_logse_min'], V['S_logse_max'] = f2(lg.min()), f2(lg.max())
H = S[S.analysis == 'mouse->human']
co = H[H['transform'] == 'coral'].rho; V['S_coral_min'], V['S_coral_max'] = f2(co.min()), f2(co.max())
V['S_n_const'] = str(int(H.rho.isna().sum())); V['S_n_tot'] = str(len(H))
V['S_n_const_sr'] = str(int(H[H['transform'].isin(['sample_z', 'rank'])].rho.isna().sum()))
bestcfg = H.groupby(['k', 'alpha', 'target'])['rho'].max()
V['S_best_min'], V['S_best_max'] = f2(bestcfg.min()), f2(bestcfg.max())
bt = H.dropna().loc[H.dropna().groupby(['k', 'alpha', 'target']).rho.idxmax()]
V['S_n_besttr'] = str(bt['transform'].nunique())
BR = pd.read_csv(O + 'S_brain_regions.csv'); ad = BR[BR['transform'] != 'raw']
cb = ad[ad.region == 'cerebellum'].rho; ot = ad[ad.region != 'cerebellum'].groupby('region').rho.max()
V['R_cb_min'], V['R_cb_max'] = f2(cb.min()), f2(cb.max()); V['R_ot_min'], V['R_ot_max'] = f2(ot.min()), f2(ot.max())
json.dump(V, open(O + 'V.json', 'w'), indent=1, ensure_ascii=False)
rng_ = H.groupby(['target', 'transform']).rho.agg(lambda x: x.max() - x.min())
V['S_maxshift'] = f2(rng_.max())
json.dump(V, open(O + 'V.json', 'w'), indent=1, ensure_ascii=False)

# cluster-bootstrap CIs of pooled mouse rho and the LOGSE/LOTO difference
PC = pd.read_csv(O + 'A_pooled_ci.csv').set_index('split')
for sp in PC.index:
    k = 'random' if sp == 'random5fold' else sp; V[f'A_lo_{k}'], V[f'A_hi_{k}'] = f2(PC.loc[sp, 'lo']), f2(PC.loc[sp, 'hi'])
# largest sensitivity shift, with its setting
_r = H.groupby(['target', 'transform']).rho.agg(lambda x: x.max() - x.min()); _t, _tr = _r.idxmax()
_x = H[(H.target == _t) & (H['transform'] == _tr)]
V['S_shift_hi'], V['S_shift_lo'] = f2(_x.rho.max()), f2(_x.rho.min())
V['S_shift_hi_k'], V['S_shift_lo_k'] = f"{int(_x.loc[_x.rho.idxmax(), 'k']):,}", f"{int(_x.loc[_x.rho.idxmin(), 'k']):,}"
json.dump(V, open(O + 'V.json', 'w'), indent=1, ensure_ascii=False)
# paired cluster-bootstrap difference LOTO - LOGSE in pooled rho
from scipy.stats import spearmanr as _sp2
_P = pd.read_csv(O + 'A_mouse_preds.csv'); _rng = np.random.default_rng(7)
_units = _P.dataset.unique(); _idx = {u: np.where(_P.dataset.values == u)[0] for u in _units}
_d = []
for _b in range(2000):
    _s = np.concatenate([_idx[u] for u in _rng.choice(_units, len(_units), replace=True)])
    _d.append(_sp2(_P.age_mo.values[_s], _P['pred_LOTO'].values[_s])[0] - _sp2(_P.age_mo.values[_s], _P['pred_LOGSE'].values[_s])[0])
V['A_diff_lo'], V['A_diff_hi'] = f2(np.percentile(_d, 2.5)), f2(np.percentile(_d, 97.5))
json.dump(V, open(O + 'V.json', 'w'), indent=1, ensure_ascii=False)

# ---- revision values
SA = pd.read_csv(O + 'A_saturation.csv'); SB = pd.read_csv(O + 'B_saturation.csv')
for sp in ['random5fold', 'LODO', 'LOGSE', 'LOTO']:
    k = 'random' if sp == 'random5fold' else sp; V[f'A_sat_{k}'] = f1(SA[(SA.split == sp) & (SA.dataset == 'ALL')].pct_bound.iloc[0])
V['A_sat_units'] = str(int(((SA.dataset != 'ALL') & (SA.pct_bound > 50) & (SA.split != 'random5fold')).sum()))
V['A_sat_units_n'] = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', '11', '12'][SA[(SA.dataset != 'ALL') & (SA.pct_bound > 50)].dataset.nunique()]
rw = SB[SB['transform'] == 'raw']; V['B_rawsat_min'], V['B_rawsat_max'] = f'{rw.pct_bound.min():.0f}', f'{rw.pct_bound.max():.0f}'
V['B_sat_cells'] = str(int((SB[SB['transform'] != 'raw'].pct_bound > 50).sum()))
_w = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
V['B_sat_sz'] = _w[int((SB[SB['transform'] == 'sample_z'].pct_bound > 50).sum())]; V['B_sat_rank'] = _w[int((SB[SB['transform'] == 'rank'].pct_bound > 50).sum())]
V['h_donors_brain'] = '144'
# best human->human per study and comparison with mouse best
for gse, t in TAG.items():
    e = E[E.test == gse]; V[f'E_{t}_best'] = f2(e.rho.max()); V[f'E_{t}_med'] = f2(e.rho.median())
V['E_n_mouse_above_best'] = str(int(sum(float(V[f'B_{t}_best']) > E[E.test == g].rho.max() for g, t in TAG.items())))
if os.path.exists(O + 'S_coral_eps.csv'):
    CE = pd.read_csv(O + 'S_coral_eps.csv')
    V['CE_shift_max'] = f2(CE.groupby('GSE').rho.agg(lambda x: x.max() - x.min()).max())
    V['CE_min_all'] = f2(CE.rho.min())
if os.path.exists(O + 'S_logse_common.csv'):
    LC = pd.read_csv(O + 'S_logse_common.csv'); V['LC_rho'] = f2(LC[LC.dataset == 'ALL'].rho.iloc[0]); V['LC_genes'] = f"{int(LC.n_genes.iloc[0]):,}"
    V['LC_medae'] = f1(LC[LC.dataset == 'ALL'].medae.iloc[0])
V['D_liv_ntest'] = str(int(NH['GSE61258']) - 20)
# descriptor analysis: distinct values
G2 = pd.read_csv(O + 'A_unit_descriptors.csv', index_col=0); V['U_n_span'] = str(G2.span.round(3).nunique()); V['U_n_cov'] = str(G2['cov'].nunique())
_w2 = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
V['U_n_span_w'], V['U_n_cov_w'] = _w2[int(V['U_n_span'])], _w2[int(V['U_n_cov'])]
json.dump(V, open(O + 'V.json', 'w'), indent=1, ensure_ascii=False)
