"""Independent residual reload for five matched fusion-seed replicas.
Cached base/CNN forecasts are used; no old raw-MAT base model replay.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import json
import numpy as np,joblib
from scipy.special import expit,logit
from threadpoolctl import threadpool_limits
import driver
from common import batches,center
from context_rank_v34 import context_features
from select_ensemble import postprocess

def main():
 threadpool_limits(1);root=driver.JOB/'seed_stability_20261101';spec=json.loads((root/'protocol.json').read_text())
 with np.load(driver.CONTROL/'full_evaluation_rows.npz') as a:mapping=a['model_row_to_full_row'];n=len(a['label'])
 reverse=np.full(n,-1,dtype=int);reverse[mapping]=np.arange(len(mapping))
 with np.load(driver.ART/'features.npz') as a:d={k:a[k] for k in ['subject','trial','crop','label','diagnosis','train','powers','quality']}
 d['label'][:]=-1;d['diagnosis'][:]=-1;g=batches(d);records=[]
 for seed in spec['seeds']:
  sub=root/f'seed{seed}'
  with np.load(sub/'baseline/subbag16_rawmean_calibrated_oof.npz') as a:teacher=a['probability']
  with np.load(sub/'residual/oof.npz') as a:expected=a['probability']
  for fold in range(1,6):
   a=joblib.load(sub/f'residual/fold{fold}.joblib');base=joblib.load(sub/f'baseline/fold{fold}.joblib')
   with np.load(driver.JOB/f'cached_group_diagnostics_20261030/fold{fold}_aligned_banks.npz') as b:va=b['outer_indices'];old=b['outer_probability'][104]
   with np.load(driver.JOB/f'neural_trial_consistency_20261031/selected_fold{fold}_outer.npz') as b:
    valid=reverse[b['validation']]>=0;assert np.array_equal(reverse[b['validation'][valid]],va);z=logit(np.clip(b['probability'][valid],1e-6,1-1e-6))
   new=postprocess(center(z,g[va])/a['scale'],g[va],'conditional',.6);delta=center(logit(np.clip(new,1e-6,1-1e-6))-logit(np.clip(old,1e-6,1-1e-6)),g[va])/a['delta_scale'];ex=context_features(d,va,base['context']);p=teacher[mapping[va]];v=np.column_stack([delta,.5*(2*ex[:,6]-1)*delta])*(4*p*(1-p))[:,None];replay=expit(logit(np.clip(p,1e-6,1-1e-6))+v@a['beta']);error=float(abs(replay-expected[mapping[va]]).max());assert error<1e-9;records.append(dict(seed=seed,fold=fold,error=error))
 result=dict(records=records,all25residual_heads_passed=True,query_labels_and_diagnoses_masked=True,old_raw_MAT_models_replayed=False)
 driver.save(root/'independent_residual_replay.json',result);print('ALL25 RESIDUAL RELOADS PASSED',max(x['error'] for x in records),flush=True)
if __name__=='__main__':main()
