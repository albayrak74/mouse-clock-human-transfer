"""Small loaders shared by the figure script (extracted from the manuscript builder)."""
import os, numpy as np, pandas as pd
O = './out/'

def load_ci():
    """Bootstrap CIs; donor-level intervals replace sample-level ones for the brain study when available."""
    CI = pd.read_csv(O + 'B_rho_ci.csv')
    if os.path.exists(O + 'B_rho_ci_donor.csv'):
        Dn = pd.read_csv(O + 'B_rho_ci_donor.csv').set_index('transform')
        for i in CI.index[CI.GSE == 'GSE15745']:
            t = CI.loc[i, 'transform']; CI.loc[i, ['rho_lo', 'rho_hi']] = Dn.loc[t, ['rho_lo', 'rho_hi']].values
    return CI

def load_E():
    f = O + 'E_human_reference_v3.csv'
    return pd.read_csv(f if os.path.exists(f) else O + 'E_human_reference_v2.csv')

def sat_pct(g, t):
    S = pd.read_csv(O + 'B_saturation.csv'); return float(S[(S.GSE == g) & (S['transform'] == t)].pct_bound.iloc[0])

def best_adaptation(g):
    CI = pd.read_csv(O + 'B_rho_ci.csv')
    c = CI[(CI.GSE == g) & (CI['transform'] != 'raw')].dropna(subset=['rho']).copy()
    c = c[[sat_pct(g, t) <= 50 for t in c['transform']]]
    return c.sort_values('rho').iloc[-1]['transform']

def human_ages(g):
    """Ages of the human samples retained in the analysis (same filters as m6_analysis.py)."""
    m = pd.read_csv(O + f'hmeta_{g}.csv', low_memory=False)
    keep = np.ones(len(m), bool)
    for col, ok in [('diseasestatus', lambda s: s.str.contains('1|control|Control|Healthy', regex=True)),
                    ('disease state', lambda s: s.str.contains('normal', case=False))]:
        if col in m:
            has = m[col].notna(); keep &= ~has.values | ok(m[col].fillna('')).values
    keep &= m.age_y.notna().values
    return m.age_y[keep]
