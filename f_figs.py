"""Figures 1-3 for Makale-9 (600 dpi PNG + TIFF)."""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O = './out/'; F = './fig/'
import os; os.makedirs(F, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b', 'xtick.color': '#52514e', 'ytick.color': '#52514e',
                     'axes.grid': True, 'grid.color': '#e6e5e0', 'grid.linewidth': 0.5, 'axes.axisbelow': True})
C = {'raw': '#8f8e89', 'sample_z': '#2a78d6', 'rank': '#eb6834', 'dataset_z': '#1baf7a', 'coral': '#eda100'}
MK = {'raw': 'o', 'sample_z': 's', 'rank': '^', 'dataset_z': 'D', 'coral': 'v'}
LAB = {'raw': 'No adaptation', 'sample_z': 'Per-sample z', 'rank': 'Within-sample rank', 'dataset_z': 'Per-dataset z', 'coral': 'CORAL'}
SPL = {'random5fold': ('Random 5-fold', '#2a78d6', 'o'), 'LODO': ('Leave-dataset-out', '#eb6834', 's'),
       'LOGSE': ('Leave-series-out', '#1baf7a', '^'), 'LOTO': ('Leave-tissue-out', '#eda100', 'D')}
HS = {'GSE40279': 'Blood, 450K (GSE40279)', 'GSE41037': 'Blood, 27K (GSE41037)', 'GSE20242': 'CD14+ monocytes, 27K (GSE20242)',
      'GSE15745': 'Brain, 27K (GSE15745)', 'GSE50498': 'Skeletal muscle, 450K (GSE50498)', 'GSE61258': 'Liver, 450K (GSE61258)'}
SH = {'GSE40279': 'Blood 450K', 'GSE41037': 'Blood 27K', 'GSE20242': 'Monocytes 27K', 'GSE15745': 'Brain 27K',
      'GSE50498': 'Muscle 450K', 'GSE61258': 'Liver 450K'}
ORDER = ['GSE40279', 'GSE41037', 'GSE20242', 'GSE15745', 'GSE50498', 'GSE61258']


def save(fig, name):
    fig.savefig(F + name + '.png', dpi=600, bbox_inches='tight', facecolor='white')
    fig.savefig(F + name + '.tif', dpi=600, bbox_inches='tight', facecolor='white', pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)


# ---------- Figure 1: within-mouse transport
A = pd.read_csv(O + 'A_mouse_transport.csv')
fig = plt.figure(figsize=(6.69, 8.2))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 2.6], wspace=0.3, hspace=0.32)
al = A[A.dataset == 'ALL'].set_index('split').loc[list(SPL)]
TICK = ['Random\n5-fold', 'Dataset\nheld out', 'Series\nheld out', 'Tissue\nheld out']
PCI = pd.read_csv(O + 'A_pooled_ci.csv').set_index('split').loc[list(SPL)]
for r, (col, ttl) in enumerate([('rho', 'Spearman ρ, all 997 mice'), ('medae', 'Median absolute error, months')]):
    ax = fig.add_subplot(gs[0, r])
    ax.bar(range(4), al[col], color=[SPL[s][1] for s in SPL], width=0.6, edgecolor='white', linewidth=1)
    if col == 'rho':
        ax.errorbar(range(4), PCI.rho, yerr=[PCI.rho - PCI.lo, PCI.hi - PCI.rho], fmt='none', ecolor='#0b0b0b', elinewidth=1, capsize=3)
        for i, v in enumerate(PCI.hi): ax.text(i, v + 0.02, f'{al[col].iloc[i]:.2f}', ha='center', va='bottom', fontsize=8)
    else:
        for i, v in enumerate(al[col]): ax.text(i, v, f'{v:.1f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(range(4)); ax.set_xticklabels(TICK, fontsize=8)
    ax.set_ylim(0, (PCI.hi.max() if col == 'rho' else al[col].max()) * 1.18); ax.tick_params(axis='x', length=0)
    ax.set_title(ttl, fontsize=8, loc='left'); ax.grid(axis='x', visible=False)
    ax.text(-0.16, 1.06, 'AB'[r], transform=ax.transAxes, fontweight='bold', fontsize=10)
ax = fig.add_subplot(gs[1, :])
d = A[A.dataset != 'ALL']
order = d[d.split == 'random5fold'].sort_values('rho').dataset.tolist()
y = {k: i for i, k in enumerate(order)}
for k in order:
    v = d[d.dataset == k].set_index('split').rho
    ax.plot([v.min(), v.max()], [y[k]] * 2, color='#c9c8c2', lw=1, zorder=1)
for s, (lab, col, mk) in SPL.items():
    v = d[d.split == s].set_index('dataset').loc[order]
    ax.scatter(v.rho, [y[k] for k in order], color=col, marker=mk, s=24, label=lab, edgecolor='white', linewidth=0.5, zorder=2)
nmap = d[d.split == 'random5fold'].set_index('dataset').n
ax.set_yticks(range(len(order))); ax.set_yticklabels([f"{k.split('_')[0]} · {k.split('_')[1]} (n={int(nmap[k])})" for k in order], fontsize=8)
ax.axvline(0, color='#52514e', lw=0.6); ax.set_xlim(-0.6, 1.0); ax.set_xlabel('Spearman ρ between predicted and chronological age')
ax.legend(loc='upper left', fontsize=8, frameon=True, framealpha=1, edgecolor='#e6e5e0')
ax.set_title('Per tissue × dataset unit', fontsize=8, loc='left'); ax.text(-0.25, 1.02, 'C', transform=ax.transAxes, fontweight='bold', fontsize=10)
save(fig, 'Figure1')

# ---------- Figure 2: mouse -> human, by transform, with human->human reference range
import sys; sys.path.insert(0, '.'); import helpers as BM
CI = BM.load_ci(); E = BM.load_E(); SAT = pd.read_csv(O + 'B_saturation.csv')
def satp(g, t): return float(SAT[(SAT.GSE == g) & (SAT['transform'] == t)].pct_bound.iloc[0])
fig, ax = plt.subplots(figsize=(6.69, 4.8))
T = list(C); seen = set()
for gi, g in enumerate(ORDER):
    e = E[(E.test == g)].rho
    if len(e): ax.add_patch(plt.Rectangle((e.min(), gi - 0.42), e.max() - e.min(), 0.84, color='#efeee9', zorder=0, lw=0))
    for ti, t in enumerate(T):
        r = CI[(CI.GSE == g) & (CI['transform'] == t)].iloc[0]; yy = gi - 0.3 + ti * 0.15
        if np.isnan(r.rho) or satp(g, t) > 50:
            ax.text(-0.39, yy, f'saturated ({satp(g, t):.0f}%)', fontsize=7, color=C[t], va='center'); continue
        ax.plot([r.rho_lo, r.rho_hi], [yy, yy], color=C[t], lw=1.2)
        ax.scatter(r.rho, yy, color=C[t], marker=MK[t], s=20, edgecolor='white', linewidth=0.5, zorder=3, label=None if t in seen else LAB[t]); seen.add(t)
ax.scatter([], [], marker='s', s=40, color='#efeee9', label='Human→human range (all five versions)')
ax.set_yticks(range(len(ORDER))); ax.set_yticklabels([HS[g] for g in ORDER]); ax.invert_yaxis()
ax.axvline(0, color='#52514e', lw=0.6); ax.set_xlim(-0.42, 1.0)
ax.set_xlabel('Spearman ρ between mouse-clock score and chronological age (95% bootstrap CI)')
ax.legend(loc='upper center', bbox_to_anchor=(0.4, -0.15), ncol=3, fontsize=8, frameon=False, handletextpad=0.3, columnspacing=0.9)
ax.grid(axis='y', visible=False)
save(fig, 'Figure2')

# ---------- Figure 3: few-shot recalibration vs k-sample mean baseline
D = pd.read_csv(O + 'D_fewshot_v2.csv')
fig, axs = plt.subplots(2, 3, figsize=(6.69, 5.6))
for ax, g in zip(axs.flat, ORDER):
    d = D[D.GSE == g]
    b = d[d['transform'] == 'dataset_z'].sort_values('k')
    ax.plot(b.k, b.mae_baseline, color='#0b0b0b', ls='--', lw=1.2, label='Mean of k labeled ages')
    for t in T:
        v = d[d['transform'] == t].sort_values('k')
        if v.mae.isna().all() or satp(g, t) > 50: continue
        if t == 'dataset_z': ax.fill_between(v.k, v.mae_lo, v.mae_hi, color=C[t], alpha=0.15, lw=0)
        ax.plot(v.k, v.mae, color=C[t], marker=MK[t], ms=3.5, lw=1.4, label=LAB[t])
    ax.set_title(SH[g], fontsize=8, loc='left'); ax.set_xscale('log'); ks = sorted(d.k.unique()); ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks])
    top = max([b.mae_baseline.max()] + [d[(d['transform'] == t)].mae.max() for t in T if satp(g, t) <= 50]); ax.set_ylim(top=top * 1.3)
    ax.minorticks_off()
for ax in axs[1]: ax.set_xlabel('Labeled target samples (k)')
for ax in axs[:, 0]: ax.set_ylabel('MAE (years)')
h, l = axs.flat[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.09, 1, 1))
save(fig, 'Figure3')
print('figures ok')
