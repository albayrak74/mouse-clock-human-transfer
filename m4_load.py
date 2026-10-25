"""Shared loaders: mouse (human-ortholog symbols) and human gene-level promoter matrices + sample tables."""
import numpy as np, pandas as pd, glob, os
O = './out/'
ORT = pd.read_csv('./ann/ortho_1to1.csv')
MM2HS = dict(zip(ORT.mm, ORT.hs))
GEST_Y = {'mouse': 19.5 / 365.25, 'human': 280 / 365.25}
MAXLS_Y = {'mouse': 4.0, 'human': 122.5}


def rel_age(age_y, sp):
    """Log relative age: log((age + gestation) / maximum lifespan)."""
    return np.log((np.asarray(age_y, float) + GEST_Y[sp]) / MAXLS_Y[sp])


def inv_rel_age(z, sp):
    return np.exp(z) * MAXLS_Y[sp] - GEST_Y[sp]


def load_mouse():
    S = pd.read_csv(O + 'samples.csv')
    S = S[S.use].copy()
    mats, metas = [], []
    for f in sorted(glob.glob(O + 'feat_*.npz')):
        z = np.load(f, allow_pickle=True)
        tag = os.path.basename(f)[5:-4]
        g = pd.Index([MM2HS.get(x) for x in z['genes']])
        X = pd.DataFrame(z['mean'].T, index=[f'{tag}|{c}' for c in z['cols']], columns=g)
        X = X.loc[:, X.columns.notna()]
        mats.append(X)
        d = S[S.file.str.endswith(f'/{tag}_count.csv')].set_index('col').loc[list(z['cols'])].reset_index()
        d['key'] = X.index; metas.append(d)
    X = pd.concat(mats).astype(np.float32); M = pd.concat(metas).set_index('key').loc[X.index]
    M['age_y'] = M.age_mo / 12.0
    M['dataset'] = M.tissue + '_' + M.GSE
    return X, M


def load_human():
    mats, metas = [], []
    for f in sorted(glob.glob(O + 'hfeat_*.npz')):
        g = os.path.basename(f)[6:-4]
        z = np.load(f, allow_pickle=True)
        V = z['mean'].T.astype(np.float64)
        V[~np.isfinite(V) | (V < -1e30)] = np.nan          # float32 missing-value sentinels (GSE20242)
        if np.nanmin(V) < -0.5:                             # stored as M-values (GSE50498) -> beta
            V = 2.0 ** V / (1.0 + 2.0 ** V)
        X = pd.DataFrame(V, index=[f'{g}|{c}' for c in z['cols']], columns=z['genes'])
        m = pd.read_csv(O + f'hmeta_{g}.csv', low_memory=False)
        m['key'] = [f'{g}|{c}' for c in m.GSM]
        mats.append(X); metas.append(m.set_index('key').loc[X.index])
    X = pd.concat(mats).astype(np.float32); M = pd.concat(metas)
    return X, M
