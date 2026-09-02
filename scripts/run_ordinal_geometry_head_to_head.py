from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from statsmodels.stats.multitest import multipletests

SOURCE_SPECIFIC_STYLES={"surrealism","ukiyo_e"}


def cols(df,p): return sorted(c for c in df.columns if c.startswith(p))

def pipeline(k=None):
    sel="passthrough" if k is None else SelectKBest(f_classif,k=int(k))
    return Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler()),("selector",sel),("clf",SVC(kernel="rbf",cache_size=2048))])

def grid(): return {"clf__C":[1.,3.,10.],"clf__gamma":["scale",.01,.03]}

def boot_idx(y,g,rng):
    ug=np.unique(g); by={u:np.flatnonzero(g==u) for u in ug}; classes=set(np.unique(y))
    for _ in range(100):
        s=rng.choice(ug,size=len(ug),replace=True); idx=np.concatenate([by[u] for u in s])
        if set(np.unique(y[idx]))==classes: return idx
    return np.arange(len(y))

def boot_ci(y,p,g,n=2000,seed=42):
    rng=np.random.default_rng(seed); v=[]
    for _ in range(n):
        i=boot_idx(y,g,rng); v.append(f1_score(y[i],p[i],average="macro",zero_division=0))
    return np.percentile(v,[2.5,97.5])

def paired(y,a,b,g,n=5000,seed=123):
    rng=np.random.default_rng(seed); v=[]
    for _ in range(n):
        i=boot_idx(y,g,rng); v.append(f1_score(y[i],a[i],average="macro",zero_division=0)-f1_score(y[i],b[i],average="macro",zero_division=0))
    obs=f1_score(y,a,average="macro",zero_division=0)-f1_score(y,b,average="macro",zero_division=0)
    return dict(delta_macro_f1=float(obs),delta_ci_low=float(np.percentile(v,2.5)),delta_ci_high=float(np.percentile(v,97.5)),bootstrap_p_new_gt_ref=float((1+np.sum(np.asarray(v)<=0))/(n+1)),n_boot=n)

def specs(df,extended):
    hc,o11,o24,o75,k,b=[cols(df,p) for p in ["ordhc__","ord11__","ord24__","ord75__","geom__curv__","base__"]]
    exp={"HC":2,"O11":11,"O24":24,"O75":75,"K":40,"B":90}; got={"HC":len(hc),"O11":len(o11),"O24":len(o24),"O75":len(o75),"K":len(k),"B":len(b)}
    if got!=exp: raise RuntimeError(f"Unexpected feature counts: {got}; expected {exp}")
    d={
      "OP_HC":(hc,None),"OP11":(o11,None),"OP24":(o24,None),"OP75":(o75,None),"OP75_k40":(o75,40),
      "OP75_K40":(sorted(o75+k),None),"OP75_K40_k40":(sorted(o75+k),40)}
    if extended:
        bo=sorted(b+o75); bok=sorted(b+o75+k)
        d.update({"B90_OP75":(bo,None),"B90_OP75_K40":(bok,None),"B90_OP75_k90":(bo,90),"B90_OP75_K40_k90":(bok,90)})
    return d

def external_preds(path,work):
    if path is None or not path.exists(): return {}
    ext=pd.read_csv(path); keys=["style","artist","filename","split"]
    wanted={"K40_curvature":"G_all_s1248","B90_strong":"B_strong","B90_K40_k90":"BG_all_s1248_kB"}
    m=work[keys].merge(ext[keys+list(wanted.values())],on=keys,how="left",validate="one_to_one")
    out={}
    for new,old in wanted.items():
        if m[old].isna().any(): raise RuntimeError(f"Could not align Phase-IVb predictions: {old}")
        out[new]=m[old].astype(str).to_numpy()
    return out

def evaluate(df,name,outdir,ext_path,extended,outer_folds,inner_folds,n_jobs,metric_boot,delta_boot):
    w=df.copy(); w["artist"]=w["artist"].fillna("").astype(str).str.strip(); w=w[w["artist"].ne("")].reset_index(drop=True)
    y=w["style"].astype(str).to_numpy(); g=w["artist"].astype(str).to_numpy(); ex=specs(w,extended)
    outer=list(StratifiedGroupKFold(outer_folds,shuffle=True,random_state=42).split(np.zeros((len(w),1)),y,groups=g))
    rows=[]; folds=[]; pred=external_preds(ext_path,w)
    for j,(ename,(fc,k)) in enumerate(ex.items(),1):
        print(f"[{name}] {j}/{len(ex)} {ename}: {len(fc)} -> {k or len(fc)}")
        oof=np.empty(len(w),dtype=object); fs=[]
        for fold,(tr,te) in enumerate(outer,1):
            inner=StratifiedGroupKFold(inner_folds,shuffle=True,random_state=100+fold)
            gs=GridSearchCV(pipeline(k),grid(),scoring="f1_macro",cv=inner,refit=True,n_jobs=n_jobs)
            gs.fit(w.iloc[tr][fc],y[tr],groups=g[tr]); pr=gs.predict(w.iloc[te][fc]); oof[te]=pr
            score=f1_score(y[te],pr,average="macro",zero_division=0); fs.append(score)
            folds.append(dict(dataset=name,experiment=ename,fold=fold,macro_f1=score,best_inner_cv_macro_f1=gs.best_score_,best_params=json.dumps(gs.best_params_,sort_keys=True)))
        pred[ename]=oof; lo,hi=boot_ci(y,oof,g,metric_boot)
        rows.append(dict(dataset=name,experiment=ename,n_features_input=len(fc),n_features_selected=(k or len(fc)),macro_f1_oof=f1_score(y,oof,average="macro",zero_division=0),macro_f1_ci_low=lo,macro_f1_ci_high=hi,fold_mean=np.mean(fs),fold_std=np.std(fs,ddof=1)))
        pd.DataFrame({**{c:w[c] for c in ["style","artist","filename","split"]},**{m:p for m,p in pred.items()}}).to_csv(outdir/f"{name}_oof_checkpoint.csv",index=False)
        pd.DataFrame(rows).to_csv(outdir/f"{name}_model_checkpoint.csv",index=False)
    for ename,p in external_preds(ext_path,w).items():
        lo,hi=boot_ci(y,p,g,metric_boot,43)
        rows.append(dict(dataset=name,experiment=ename,n_features_input=(40 if ename=="K40_curvature" else 90),n_features_selected=(40 if ename=="K40_curvature" else 90),macro_f1_oof=f1_score(y,p,average="macro",zero_division=0),macro_f1_ci_low=lo,macro_f1_ci_high=hi,fold_mean=np.nan,fold_std=np.nan))
    comps=[("OP75","OP_HC","ordinal_75_vs_HC",False),("OP75","OP11","ordinal_75_vs_11",False),("OP75","OP24","ordinal_75_vs_24",False),("OP75_K40","OP75","curvature_increment_over_ordinal_full",True),("OP75_K40_k40","OP75_k40","curvature_increment_over_ordinal_k40",True),("K40_curvature","OP75_k40","curvature_vs_ordinal_k40",False)]
    if extended: comps += [("B90_OP75","B90_strong","ordinal_increment_over_B",False),("B90_OP75_K40","B90_OP75","curvature_increment_over_B_plus_ordinal_full",True),("B90_OP75_K40_k90","B90_OP75_k90","curvature_increment_over_B_plus_ordinal_k90",True),("B90_OP75_K40_k90","B90_K40_k90","ordinal_increment_over_B_plus_curvature_k90",False)]
    ds=[]
    for new,ref,label,primary in comps:
        if new in pred and ref in pred:
            d=paired(y,pred[new],pred[ref],g,delta_boot); d.update(dataset=name,contrast=label,new_model=new,reference=ref,primary_head_to_head=primary); ds.append(d)
    ddf=pd.DataFrame(ds)
    if len(ddf):
        mask=ddf.primary_head_to_head.astype(bool); ddf["bootstrap_q_bh_primary"]=np.nan
        if mask.any(): ddf.loc[mask,"bootstrap_q_bh_primary"]=multipletests(ddf.loc[mask,"bootstrap_p_new_gt_ref"],method="fdr_bh")[1]
    oof=pd.DataFrame({**{c:w[c] for c in ["style","artist","filename","split"]},**{m:p for m,p in pred.items()}})
    return pd.DataFrame(rows),ddf,pd.DataFrame(folds),oof

def main(features,outdir,all_oof,w8_oof,outer,inner,n_jobs,metric_boot,delta_boot):
    outdir.mkdir(parents=True,exist_ok=True); df=pd.read_csv(features); df["style"]=df["style"].astype(str)
    configs=[("artbench10_all",df,all_oof,False),("artbench10_wikiart8",df[~df["style"].isin(SOURCE_SPECIFIC_STYLES)].reset_index(drop=True),w8_oof,True)]
    rr=[]; dd=[]; ff=[]
    for name,sub,ext,extended in configs:
        r,d,f,o=evaluate(sub,name,outdir,ext,extended,outer,inner,n_jobs,metric_boot,delta_boot); rr.append(r); dd.append(d); ff.append(f); o.to_csv(outdir/f"{name}_phase6_oof_predictions.csv",index=False)
    R=pd.concat(rr,ignore_index=True); D=pd.concat(dd,ignore_index=True); F=pd.concat(ff,ignore_index=True)
    R.to_csv(outdir/"phase6_head_to_head_results.csv",index=False); D.to_csv(outdir/"phase6_head_to_head_deltas.csv",index=False); F.to_csv(outdir/"phase6_head_to_head_fold_results.csv",index=False)
    print("\nRESULTS\n",R.sort_values(["dataset","macro_f1_oof"],ascending=[True,False]).to_string(index=False)); print("\nDELTAS\n",D.to_string(index=False))

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--features",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--phase4b-all-oof",type=Path); p.add_argument("--phase4b-wiki8-oof",type=Path); p.add_argument("--outer-folds",type=int,default=5); p.add_argument("--inner-folds",type=int,default=3); p.add_argument("--n-jobs",type=int,default=-1); p.add_argument("--metric-boot",type=int,default=2000); p.add_argument("--delta-boot",type=int,default=5000); a=p.parse_args(); main(a.features,a.output_dir,a.phase4b_all_oof,a.phase4b_wiki8_oof,a.outer_folds,a.inner_folds,a.n_jobs,a.metric_boot,a.delta_boot)
