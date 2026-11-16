"""Paired branch/fusion ablation for the cached two-second BMRC LGO protocol."""
from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from evidential import fit_model
from run_bmrc_leave_one_generator import (
    ROOT, OUT, SEEDS, BENCHMARKS, load_rows, split_group, stable_seed,
)

OUT_ABL = Path(os.environ.get("BMRC_ABLATION_OUTPUT", ROOT / "results/component_ablation"))

def metric(y, s):
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))

def main():
    torch.set_num_threads(2)
    OUT_ABL.mkdir(parents=True, exist_ok=True)
    d = load_rows()
    rows=[]
    for bench in BENCHMARKS:
        bi=np.flatnonzero(d['benchmark']==bench)
        gens=sorted(set(d['generator'][bi][d['is_real'][bi]==0].tolist()))
        if len(gens)<2: continue
        real_parts={}
        for source in sorted(set(d['generator'][bi][d['is_real'][bi]==1].tolist())):
            ix=bi[(d['is_real'][bi]==1)&(d['generator'][bi]==source)]
            real_parts[source]=split_group(ix,f'{bench}:real:{source}')
        for target in gens:
            target_fake=bi[(d['is_real'][bi]==0)&(d['generator'][bi]==target)]
            known=[g for g in gens if g!=target]
            fake_parts={g:split_group(bi[(d['is_real'][bi]==0)&(d['generator'][bi]==g)],f'{bench}:fake:{g}') for g in known}
            tr=np.concatenate([p['train'] for p in real_parts.values()]+[p['train'] for p in fake_parts.values()])
            va=np.concatenate([p['validation'] for p in real_parts.values()]+[p['validation'] for p in fake_parts.values()])
            real_test=np.concatenate([p['test'] for p in real_parts.values()])
            n=min(len(real_test),len(target_fake))
            rng=np.random.default_rng(20261112+stable_seed(f'{bench}:{target}:balanced'))
            te=np.sort(np.concatenate([rng.choice(real_test,n,replace=False),rng.choice(target_fake,n,replace=False)]))
            y=d['is_real'][te]
            for seed in SEEDS:
                model,_=fit_model(d['features'][tr],d['percentiles'][tr],d['quality'][tr],d['is_real'][tr],
                                  d['features'][va],d['percentiles'][va],d['quality'][va],d['is_real'][va],
                                  seed=seed,epochs=150,objective='mse')
                with torch.no_grad():
                    raw=model(torch.as_tensor(d['features'][te],dtype=torch.float32))
                    residual=np.stack([branch[0].numpy() for branch in raw],axis=1)
                p=d['percentiles'][te]
                corrected=np.clip(p+residual,0,1)
                q=d['quality'][te]
                w=q/np.maximum(q.sum(axis=1,keepdims=True),1e-12)
                scores={
                    'BMRC_full_quality_gate':(w*corrected).sum(axis=1),
                    'BMRC_equal_corrected_fusion':corrected.mean(axis=1),
                    'BMRC_spatial_head_only':corrected[:,0],
                    'BMRC_temporal_head_only':corrected[:,1],
                    'STALL_fixed_baseline':d['stall'][te],
                }
                for method,score in scores.items():
                    auc,ap=metric(y,score)
                    rows.append({'benchmark':bench,'generator':target,'seed':seed,'method':method,
                                 'n_each_class':n,'auroc':auc,'ap_real':ap})
            print(f'completed {bench}/{target}',flush=True)
    frame=pd.DataFrame(rows)
    frame.to_csv(OUT_ABL/'per_seed_metrics.csv',index=False)
    per=frame.groupby(['benchmark','generator','method'],as_index=False).agg(
        auroc_mean=('auroc','mean'),auroc_sd=('auroc','std'),ap_real_mean=('ap_real','mean'),
        ap_real_sd=('ap_real','std'),n_each_class=('n_each_class','first'))
    per.to_csv(OUT_ABL/'per_generator_summary.csv',index=False)
    macros=per.groupby(['benchmark','method'],as_index=False).agg(
        auroc=('auroc_mean','mean'),ap_real=('ap_real_mean','mean'),groups=('generator','count'))
    macros.to_csv(OUT_ABL/'benchmark_macro.csv',index=False)
    overall=per.groupby('method').agg(auroc=('auroc_mean','mean'),ap_real=('ap_real_mean','mean'),groups=('generator','count')).reset_index()
    overall.to_csv(OUT_ABL/'overall_macro.csv',index=False)
    (OUT_ABL/'protocol.json').write_text(json.dumps({
        'description':'Paired two-second BMRC component ablation on identical leave-one-generator-out splits and test IDs.',
        'seeds':[42,43,44],
        'methods':['STALL fixed baseline','BMRC spatial residual only','BMRC temporal residual only','BMRC equal corrected-score fusion','BMRC full quality-gated fusion'],
        'metrics':'AUROC and AP with real clips positive; average over optimization seeds within generator, then equal-generator macro.',
        'limitations':['Three seeds reuse the same held-out clips and are not independent data replicates.','Component variants are diagnostic ablations on this cache and do not establish out-of-domain transfer.']
    },indent=2))
    print(overall.to_string(index=False))

if __name__=='__main__': main()
