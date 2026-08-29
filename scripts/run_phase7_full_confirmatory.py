from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

CGRID = [0.03, 0.3, 3.0]
EXCLUDE = {'surrealism', 'ukiyo_e'}

def bh(p):
    p=np.asarray(p,float); o=np.argsort(p); r=p[o]
    q=r*len(p)/np.arange(1,len(p)+1); q=np.minimum.accumulate(q[::-1])[::-1]
    z=np.empty_like(q); z[o]=np.clip(q,0,1); return z

def scols(df,s):
    t=f'_s{float(s):.1f}'.replace('.','p')
    return [c for c in df if c.startswith('geom__curv__') and t in c]

def reps(df):
    B=[c for c in df if c.startswith('base__')]
    K=[c for c in df if c.startswith('geom__curv__')]
    O=[c for c in df if c.startswith('geom__orient__')]
    OP=[c for c in df if c.startswith('ord75__')]
    S={s:scols(df,s) for s in [1,2,4,8]}
    R={'B90':B,'K40':K,'G44':K+O,'B90_K40':B+K,'B90_G44':B+K+O,
       'OP75':OP,'OP75_K40':OP+K,'B90_OP75':B+OP,'B90_OP75_K40':B+OP+K,
       'K_s1':S[1],'K_s2':S[2],'K_s4':S[4],'K_s8':S[8],
       'K_fine_s1_s2':S[1]+S[2],'K_coarse_s4_s8':S[4]+S[8],'K_all':K}
    return {k:v for k,v in R.items() if v}

def pipe(C):
    return Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler()),
                     ('clf',LinearSVC(C=C,dual='auto',max_iter=20000,random_state=42))])

def pickC(X,y,g,tr,inner=3):
    sp=StratifiedGroupKFold(n_splits=inner,shuffle=True,random_state=4201)
    vals=[]
    for C in CGRID:
        z=[]
        for a,b in sp.split(X[tr],y[tr],groups=g[tr]):
            m=pipe(C); m.fit(X[tr][a],y[tr][a])
            z.append(f1_score(y[tr][b],m.predict(X[tr][b]),average='macro'))
        vals.append((np.mean(z),C))
    return max(vals,key=lambda x:x[0])[1], vals

def boot(y,a,b,g,n=5000,seed=42):
    rng=np.random.default_rng(seed); u=np.unique(g); idx={x:np.flatnonzero(g==x) for x in u}
    obs=f1_score(y,a,average='macro')-f1_score(y,b,average='macro'); v=[]
    for _ in range(n):
        ii=np.concatenate([idx[x] for x in rng.choice(u,len(u),replace=True)])
        v.append(f1_score(y[ii],a[ii],average='macro')-f1_score(y[ii],b[ii],average='macro'))
    v=np.asarray(v); lo,hi=np.quantile(v,[.025,.975]); p=(1+(v<=0).sum())/(n+1)
    return obs,lo,hi,p

def run(df,label,out,outer=5,inner=3,nboot=5000):
    d=df.copy(); d['artist']=d.artist.fillna('').astype(str).str.strip(); d=d[d.artist.ne('')].reset_index(drop=True)
    y=d.style.astype(str).to_numpy(); g=d.artist.to_numpy(); R=reps(d)
    sp=StratifiedGroupKFold(n_splits=outer,shuffle=True,random_state=20260829)
    splits=list(sp.split(np.zeros((len(d),1)),y,groups=g))
    preds=d[['split','style','artist','filename']].copy(); rows=[]; fr=[]
    for name,cols in R.items():
        print(label,name,len(cols),flush=True); X=d[cols].to_numpy(np.float32); P=np.empty(len(d),object); F=np.full(len(d),-1)
        for k,(tr,te) in enumerate(splits):
            C,search=pickC(X,y,g,tr,inner); m=pipe(C); m.fit(X[tr],y[tr]); p=m.predict(X[te]); P[te]=p; F[te]=k
            fr.append(dict(dataset=label,representation=name,fold=k,C=C,n_train=len(tr),n_test=len(te),
                           macro_f1=f1_score(y[te],p,average='macro'),accuracy=accuracy_score(y[te],p),inner_search=json.dumps(search)))
        preds['pred__'+name]=P.astype(str); preds['outer_fold']=F
        rows.append(dict(dataset=label,representation=name,n_features=len(cols),n_images=len(d),n_artists=d.artist.nunique(),
                         macro_f1=f1_score(y,P.astype(str),average='macro'),accuracy=accuracy_score(y,P.astype(str))))
    out.mkdir(parents=True,exist_ok=True); res=pd.DataFrame(rows)
    res.to_csv(out/'linear_probe_results.csv',index=False); pd.DataFrame(fr).to_csv(out/'linear_probe_fold_results.csv',index=False)
    preds.to_csv(out/'linear_probe_predictions.csv',index=False)
    H=[('B90_K40','B90','H1 curvature beyond appearance'),('B90_G44','B90','H1b full geometry beyond appearance'),
       ('OP75_K40','OP75','H2 curvature beyond ordinal'),('B90_OP75_K40','B90_OP75','H3 curvature beyond appearance+ordinal')]
    dr=[]
    for i,(a,b,h) in enumerate(H):
        if 'pred__'+a in preds and 'pred__'+b in preds:
            z=boot(y,preds['pred__'+a].to_numpy(),preds['pred__'+b].to_numpy(),g,nboot,42+i)
            dr.append(dict(dataset=label,hypothesis=h,new_model=a,reference=b,delta_macro_f1=z[0],ci_low=z[1],ci_high=z[2],p_one_sided=z[3],n_boot=nboot))
    D=pd.DataFrame(dr)
    if len(D): D['q_bh']=bh(D.p_one_sided.to_numpy())
    D.to_csv(out/'confirmatory_deltas.csv',index=False)
    scale=['K_s1','K_s2','K_s4','K_s8','K_fine_s1_s2','K_coarse_s4_s8','K_all']
    res[res.representation.isin(scale)].to_csv(out/'scale_probe_results.csv',index=False)
    return res,D

def main(inp,out,outer,inner,nboot):
    df=pd.read_csv(inp); allr=[]; alld=[]
    for d,l in [(df,'artbench10_all'),(df[~df.style.astype(str).isin(EXCLUDE)].copy(),'artbench10_wikiart8')]:
        r,z=run(d,l,out/l,outer,inner,nboot); allr.append(r); alld.append(z)
    pd.concat(allr).to_csv(out/'phase7_confirmatory_results_all.csv',index=False)
    pd.concat(alld).to_csv(out/'phase7_confirmatory_deltas_all.csv',index=False)
    (out/'analysis_plan.json').write_text(json.dumps({'classifier':'StandardScaler + LinearSVC','C_grid':CGRID,
        'outer_folds':outer,'inner_folds':inner,'n_boot':nboot,'group':'artist','metric':'macro-F1'},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--features',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--outer-folds',type=int,default=5); p.add_argument('--inner-folds',type=int,default=3); p.add_argument('--n-boot',type=int,default=5000)
    a=p.parse_args(); main(a.features,a.output_dir,a.outer_folds,a.inner_folds,a.n_boot)
