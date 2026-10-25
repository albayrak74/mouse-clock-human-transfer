"""Clock fitting with optional domain-adaptation feature transforms."""
import numpy as np, pandas as pd
from sklearn.linear_model import ElasticNet
from scipy.stats import spearmanr, pearsonr

TRANSFORMS = ['raw', 'sample_z', 'rank', 'dataset_z', 'coral']


def _impute(X, med):
    X = X.copy()
    m = np.isnan(X)
    X[m] = np.take(med, np.where(m)[1])
    return X


def _sample_z(X):
    mu = X.mean(1, keepdims=True); sd = X.std(1, keepdims=True) + 1e-6
    return (X - mu) / sd


def _rank(X):
    r = np.argsort(np.argsort(X, axis=1), axis=1).astype(np.float32)
    return r / (X.shape[1] - 1) - 0.5


def _group_z(X, groups):
    Z = np.empty_like(X)
    for g in np.unique(groups):
        i = groups == g
        mu = X[i].mean(0); sd = X[i].std(0) + 1e-3
        Z[i] = (X[i] - mu) / sd
    return Z


def _coral(Xs, Xt, eps=1.0):
    """CORAL: whiten source, recolour with target covariance (Sun et al. 2016); features already z-scored."""
    Cs = np.cov(Xs, rowvar=False) + eps * np.eye(Xs.shape[1])
    Ct = np.cov(Xt, rowvar=False) + eps * np.eye(Xt.shape[1])
    def msqrt(C, inv=False):
        w, V = np.linalg.eigh(C); w = np.clip(w, 1e-8, None)
        return (V * (w ** (-0.5 if inv else 0.5))) @ V.T
    return Xs @ msqrt(Cs, True) @ msqrt(Ct)


def select_genes(Xtr, ytr, k):
    """Top-k genes by |Spearman| with age in the training data (computed on ranks for speed)."""
    R = np.apply_along_axis(lambda c: pd.Series(c).rank().values, 0, Xtr)
    yr = pd.Series(ytr).rank().values
    R = (R - R.mean(0)) / (R.std(0) + 1e-9); yr = (yr - yr.mean()) / yr.std()
    rho = (R * yr[:, None]).mean(0)
    return np.argsort(-np.abs(rho))[:k], rho


def fit_predict(Xtr, ytr, gtr, Xte, gte, transform='raw', k=1000, alpha=0.01, l1=0.5, coverage=0.7):
    """Xtr/Xte: DataFrames with identical gene columns; gtr/gte: dataset labels (used by dataset_z)."""
    Xtr = Xtr.replace([np.inf, -np.inf], np.nan); Xte = Xte.replace([np.inf, -np.inf], np.nan)
    ok = (Xtr.notna().mean() >= coverage) & (Xte.notna().mean() >= coverage)
    Xtr, Xte = Xtr.loc[:, ok], Xte.loc[:, ok]
    med = np.nanmedian(Xtr.values, 0)
    A, B = _impute(Xtr.values, med), _impute(Xte.values, np.where(np.isnan(np.nanmedian(Xte.values, 0)), med, np.nanmedian(Xte.values, 0)))
    if transform == 'dataset_z' or transform == 'coral':
        A, B = _group_z(A, np.asarray(gtr)), _group_z(B, np.asarray(gte))
    idx, _ = select_genes(A, ytr, min(k, A.shape[1]))
    A, B = A[:, idx], B[:, idx]
    if transform == 'sample_z': A, B = _sample_z(A), _sample_z(B)
    if transform == 'rank': A, B = _rank(A), _rank(B)
    if transform == 'coral': A = _coral(A, B)
    A, B = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0), np.nan_to_num(B, nan=0.0, posinf=0.0, neginf=0.0)
    mu, sd = A.mean(0), A.std(0) + 1e-6
    m = ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=20000).fit((A - mu) / sd, ytr)
    pred = m.predict((B - mu) / sd)
    # keep predictions inside a biologically possible range: from just before birth to the maximum lifespan
    pred = np.clip(pred, np.min(ytr) - 0.5, 0.0)
    return pred, np.asarray(Xtr.columns)[idx], m


def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    out = {'n': len(y), 'mae': np.mean(np.abs(p - y)), 'medae': np.median(np.abs(p - y)), 'bias': np.mean(p - y)}
    if np.std(y) > 0 and np.std(p) > 0:
        out['r'] = pearsonr(y, p)[0]; out['rho'] = spearmanr(y, p)[0]
        out['slope'] = np.polyfit(y, p, 1)[0]
    else:
        out['r'] = out['rho'] = out['slope'] = np.nan
    return out
