from __future__ import annotations

from pathlib import Path
import argparse
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Patch


COLORS = {
    'B90': '#365c8d', 'K40': '#4c78a8', 'G44': '#54a24b', 'OP75': '#b279a2',
    'B90_K40': '#1f77b4', 'B90_G44': '#2ca02c', 'OP75_K40': '#9467bd',
    'B90_OP75': '#8c564b', 'B90_OP75_K40': '#d62728',
    'all10': '#1f4e79', 'wiki8': '#8c2d04',
    'style': '#4c78a8', 'artist': '#f58518', 'residual': '#bdbdbd',
}
DATASET_LABELS = {
    'artbench10_all': 'ArtBench-10',
    'artbench10_wikiart8': 'WikiArt-8 control',
}
REP_LABELS = {
    'B90': 'Appearance (B90)', 'K40': 'Curvature (K40)', 'G44': 'Full geometry (G44)',
    'OP75': 'Ordinal patterns (OP75)', 'B90_K40': 'B90 + K40', 'B90_G44': 'B90 + G44',
    'OP75_K40': 'OP75 + K40', 'B90_OP75': 'B90 + OP75',
    'B90_OP75_K40': 'B90 + OP75 + K40',
}


def setup_style():
    plt.rcParams.update({
        'figure.dpi': 140, 'savefig.dpi': 400,
        'font.family': 'DejaVu Sans', 'font.size': 10.2,
        'axes.titlesize': 11.5, 'axes.labelsize': 10.2,
        'xtick.labelsize': 9.0, 'ytick.labelsize': 9.0,
        'legend.fontsize': 8.8, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.linewidth': 0.8,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })


def find_latest(root: Path, name: str) -> Path:
    hits = [p for p in root.rglob(name) if p.is_file()]
    if not hits:
        raise FileNotFoundError(f'Could not locate {name} below {root}')
    return max(hits, key=lambda p: p.stat().st_mtime)


def panel(ax, label):
    ax.text(-0.12, 1.06, label, transform=ax.transAxes,
            fontsize=13.5, fontweight='bold', va='bottom')


def grid_y(ax):
    ax.grid(axis='y', color='#d9d9d9', lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def save(fig, outdir: Path, stem: str):
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(outdir / f'{stem}.{ext}', bbox_inches='tight', facecolor='white')


def load_inputs(drive_root: Path):
    phase7_results = pd.read_csv(find_latest(drive_root, 'phase7_fixed_results_all.csv'))
    phase7_deltas = pd.read_csv(find_latest(drive_root, 'phase7_fixed_deltas_all.csv'))
    phase5 = pd.read_csv(find_latest(drive_root, 'phase5_scale_summary.csv'))
    features_path = find_latest(drive_root, 'artbench_full_features_with_ordinal.csv')
    return phase7_results, phase7_deltas, phase5, features_path


def figure2(res: pd.DataFrame, delt: pd.DataFrame, outdir: Path):
    reps = ['B90','K40','G44','OP75','B90_K40','B90_G44','OP75_K40','B90_OP75','B90_OP75_K40']
    fig = plt.figure(figsize=(14.2, 8.3))
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.05, 0.95], hspace=0.48, wspace=0.24)

    for j, ds in enumerate(['artbench10_all','artbench10_wikiart8']):
        ax = fig.add_subplot(gs[0, j]); panel(ax, chr(ord('a')+j))
        s = res[(res.dataset == ds) & (res.representation.isin(reps))].copy()
        s['ord'] = s.representation.map({r:i for i,r in enumerate(reps)})
        s = s.sort_values('ord')
        x = np.arange(len(s))
        vals = s.macro_f1.to_numpy()
        cols = [COLORS.get(r, '#555555') for r in s.representation]
        ax.bar(x, vals, color=cols, edgecolor='black', lw=0.45)
        ax.set_xticks(x)
        ax.set_xticklabels([REP_LABELS.get(r,r) for r in s.representation], rotation=34, ha='right')
        ax.set_ylabel('Macro-F1')
        ax.set_title(DATASET_LABELS[ds], loc='left', fontweight='bold')
        grid_y(ax)
        ax.set_ylim(max(0.15, vals.min()-0.04), min(0.46, vals.max()+0.035))

    ax = fig.add_subplot(gs[1,:]); panel(ax, 'c')
    specs = [
        ('B90_K40','B90','B90 + K40 vs B90'),
        ('B90_G44','B90','B90 + G44 vs B90'),
        ('OP75_K40','OP75','OP75 + K40 vs OP75'),
        ('B90_OP75_K40','B90_OP75','B90 + OP75 + K40 vs B90 + OP75'),
    ]
    rows = []
    for ds in ['artbench10_all','artbench10_wikiart8']:
        for new, ref, lab in specs:
            h = delt[(delt.dataset==ds) & (delt.new_model==new) & (delt.reference==ref)]
            if len(h):
                r = h.iloc[0].copy(); r['label']=lab; rows.append(r)
    f = pd.DataFrame(rows).reset_index(drop=True)
    ypos = np.arange(len(f))[::-1]
    for i,r in f.iterrows():
        c = COLORS['all10'] if r.dataset=='artbench10_all' else COLORS['wiki8']
        ax.hlines(ypos[i], r.ci_low, r.ci_high, color=c, lw=2.2)
        ax.plot(r.delta_macro_f1, ypos[i], 'o', ms=6, color=c)
        ax.text(r.ci_high+0.002, ypos[i], f"Δ={r.delta_macro_f1:.4f}; q={r.q_bh:.4f}", va='center', fontsize=8.5)
    ax.axvline(0, color='black', ls='--', lw=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{DATASET_LABELS[r.dataset]} — {r['label']}" for _,r in f.iterrows()])
    ax.set_xlabel('Increment in macro-F1 (95% artist-bootstrap CI)')
    grid_y(ax)
    ax.set_xlim(min(-0.005, f.ci_low.min()-0.008), f.ci_high.max()+0.035)

    fig.suptitle('Complementarity under unseen-artist generalisation', x=0.02, y=0.995,
                 ha='left', fontsize=16, fontweight='bold')
    save(fig, outdir, 'Figure2_complementarity_hssc')
    plt.close(fig)


def figure3(res: pd.DataFrame, outdir: Path):
    scale_order = ['K_s1','K_s2','K_s4','K_s8','K_fine_s1_s2','K_coarse_s4_s8','K_all']
    labels = ['σ = 1','σ = 2','σ = 4','σ = 8','fine (σ1+σ2)','coarse (σ4+σ8)','all scales']
    fig, axes = plt.subplots(1,2, figsize=(13.6,5.4), constrained_layout=True)
    for k, ax in enumerate(axes): panel(ax, chr(ord('a')+k))

    x = np.arange(len(scale_order)); w=0.36
    for j,ds in enumerate(['artbench10_all','artbench10_wikiart8']):
        s = res[(res.dataset==ds) & (res.representation.isin(scale_order))].copy()
        vals = [float(s[s.representation==r].macro_f1.iloc[0]) for r in scale_order]
        c = COLORS['all10'] if ds=='artbench10_all' else COLORS['wiki8']
        axes[0].bar(x+(j-0.5)*w, vals, width=w, color=c, edgecolor='black', lw=0.45,
                    label=DATASET_LABELS[ds])
        axes[1].plot(x, vals, marker='o', lw=2.2, color=c, label=DATASET_LABELS[ds])
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=23, ha='right'); ax.set_ylabel('Macro-F1'); grid_y(ax)
    axes[0].set_title('Artist-disjoint classification by spatial scale', loc='left', fontweight='bold')
    axes[1].set_title('Intermediate geometry is most discriminative', loc='left', fontweight='bold')
    axes[0].legend(frameon=False); axes[1].legend(frameon=False, loc='lower right')
    fig.suptitle('Multiscale discrimination across spatial scales', x=0.02, y=1.04,
                 ha='left', fontsize=16, fontweight='bold')
    save(fig, outdir, 'Figure3_scale_discrimination_hssc')
    plt.close(fig)


def figure4(p5: pd.DataFrame, outdir: Path):
    # Reserve a dedicated lower band for the shared legend.  A figure-level
    # legend outside constrained_layout can otherwise overlap the sigma tick labels.
    fig, axes = plt.subplots(2,2, figsize=(13.7,9.7), constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.14, top=0.91,
                        hspace=0.22, wspace=0.16)
    for lab,ax in zip('abcd', axes.ravel()): panel(ax, lab)
    sigmas=[1.0,2.0,4.0,8.0]; xl=['σ = 1','σ = 2','σ = 4','σ = 8']; x=np.arange(4)
    for ds in ['artbench10_all','artbench10_wikiart8']:
        s=p5[p5.dataset==ds].set_index('sigma_ref').loc[sigmas]
        c=COLORS['all10'] if ds=='artbench10_all' else COLORS['wiki8']
        axes[0,0].plot(x,s.eta2_style_artist_centroids,marker='o',lw=2.2,color=c,label=DATASET_LABELS[ds])
        axes[0,1].plot(x,s.style_share_of_between_artist_variation,marker='o',lw=2.2,color=c,label=DATASET_LABELS[ds])
    axes[0,0].set_ylabel(r'$\eta^2_{style}$ on artist centroids')
    axes[0,1].set_ylabel('Style share of between-artist variation')
    axes[0,0].set_title('Style organisation peaks at intermediate scale',loc='left',fontweight='bold')
    axes[0,1].set_title('Style is a modest component of artist geometry',loc='left',fontweight='bold')
    for ax in axes[0]:
        ax.set_xticks(x); ax.set_xticklabels(xl); ax.legend(frameon=False); grid_y(ax)

    for col,ds in enumerate(['artbench10_all','artbench10_wikiart8']):
        ax=axes[1,col]
        s=p5[p5.dataset==ds].set_index('sigma_ref').loc[sigmas]
        a=s.style_fraction.to_numpy(); b=s.artist_within_style_fraction.to_numpy(); c=s.painting_residual_fraction.to_numpy()
        ax.bar(x,a,color=COLORS['style'],edgecolor='black',lw=.4,label='Style')
        ax.bar(x,b,bottom=a,color=COLORS['artist'],edgecolor='black',lw=.4,label='Artist within style')
        ax.bar(x,c,bottom=a+b,color=COLORS['residual'],edgecolor='black',lw=.4,label='Residual / painting-level')
        ax.set_xticks(x); ax.set_xticklabels(xl); ax.set_ylim(0,1.03); ax.set_ylabel('Fraction of geometric variation')
        ax.set_title(f"Variance decomposition — {DATASET_LABELS[ds]}",loc='left',fontweight='bold')
        grid_y(ax)

    # Shared legend centred inside its own reserved band below both lower axes.
    variance_handles = [
        Patch(facecolor=COLORS['style'], edgecolor='black', linewidth=.4, label='Style'),
        Patch(facecolor=COLORS['artist'], edgecolor='black', linewidth=.4, label='Artist within style'),
        Patch(facecolor=COLORS['residual'], edgecolor='black', linewidth=.4, label='Residual / painting-level'),
    ]
    fig.legend(handles=variance_handles, loc='lower center', bbox_to_anchor=(0.5, 0.025),
               ncol=3, frameon=False, columnspacing=2.0, handlelength=1.8)

    fig.suptitle('Style organisation in multiscale geometric space', x=0.02, y=0.975,
                 ha='left', fontsize=16, fontweight='bold')
    save(fig,outdir,'Figure4_style_organisation_hssc')
    plt.close(fig)


def tables(res: pd.DataFrame, delt: pd.DataFrame, features_path: Path, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    feat = pd.read_csv(features_path, usecols=lambda c: c in {'style','artist','filename','split'}, low_memory=False)
    feat['artist'] = feat['artist'].fillna('').astype(str).str.strip()
    linked = feat.artist.ne('')
    w8 = ~feat['style'].astype(str).isin(['surrealism','ukiyo_e'])
    t1 = pd.DataFrame([
        ['ArtBench-10', int(feat.style.nunique()), len(feat), int(linked.sum()), int(feat.loc[linked,'artist'].nunique()), 'Artist-disjoint 5-fold', 'Primary confirmatory corpus'],
        ['WikiArt-8 control', int(feat.loc[w8,'style'].nunique()), int(w8.sum()), int((linked&w8).sum()), int(feat.loc[linked&w8,'artist'].nunique()), 'Artist-disjoint 5-fold', 'Source-composition sensitivity'],
    ], columns=['Dataset','Styles','Images','Artist-linked images','Artists','Protocol','Role'])
    t1.to_csv(outdir/'Table1_dataset_protocol_summary.csv',index=False)
    (outdir/'Table1_dataset_protocol_summary.tex').write_text(t1.to_latex(index=False,escape=False),encoding='utf-8')

    specs=[('B90_K40','B90','Appearance + curvature vs appearance'),('B90_G44','B90','Appearance + full geometry vs appearance'),('OP75_K40','OP75','Ordinal + curvature vs ordinal'),('B90_OP75_K40','B90_OP75','Appearance + ordinal + curvature vs appearance + ordinal')]
    rows=[]
    for ds in ['artbench10_all','artbench10_wikiart8']:
        for new,ref,label in specs:
            r=delt[(delt.dataset==ds)&(delt.new_model==new)&(delt.reference==ref)].iloc[0]
            rows.append([DATASET_LABELS[ds],label,r.delta_macro_f1,f"[{r.ci_low:.4f}, {r.ci_high:.4f}]",r.p_one_sided,r.q_bh])
    t2=pd.DataFrame(rows,columns=['Dataset','Contrast','Delta macro-F1','95% CI','One-sided p','BH q'])
    t2.to_csv(outdir/'Table2_confirmatory_contrasts.csv',index=False)
    (outdir/'Table2_confirmatory_contrasts.tex').write_text(t2.to_latex(index=False,escape=False),encoding='utf-8')


def main(drive_root: Path, output_root: Path):
    setup_style()
    res,delt,p5,features_path=load_inputs(drive_root)
    figdir=output_root/'figures'; tabdir=output_root/'tables'
    figure2(res,delt,figdir); figure3(res,figdir); figure4(p5,figdir); tables(res,delt,features_path,tabdir)
    package=output_root/'HSSC_main_figures_tables.zip'
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED) as zf:
        for p in output_root.rglob('*'):
            if p.is_file() and p!=package:
                zf.write(p,arcname=p.relative_to(output_root))
    print('HSSC figure/table build complete ✓')
    print('Output:', output_root)
    print('Package:', package)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--drive-root',type=Path,default=Path('/content/drive/MyDrive'))
    ap.add_argument('--output-root',type=Path,default=Path('/content/drive/MyDrive/painting_geometry_phase7_full/paper_hssc'))
    a=ap.parse_args(); main(a.drive_root,a.output_root)
