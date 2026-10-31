"""Native fusion-head replay on held-subject cached base/CNN forecasts.
No new predictions on official test; metrics use original unweighted pooling.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import json,hashlib
from pathlib import Path
import numpy as np,joblib
from scipy.special import logit,expit
from threadpoolctl import threadpool_limits
import driver
from common import batches,center
from cached_fixed_fusion_20261030 import features
from context_rank_v34 import context_features
from select_ensemble import postprocess
from inner_calibration_round import calibrate

def main():
 threadpool_limits(1);root=driver.JOB/'adaptive_consistency_mix_20261101'
 with np.load(driver.ART/'features.npz') as a:d={k:a[k] for k in ['subject','trial','crop','label','diagnosis','train','powers','quality']}
 d['label'][:]=-1;d['diagnosis'][:]=-1;g=batches(d)
 with np.load(driver.CONTROL/'full_evaluation_rows.npz') as a:full={k:a[k] for k in a.files}
 reverse=np.full(len(full['label']),-1,dtype=int);reverse[full['model_row_to_full_row']]=np.arange(len(d['subject']))
 with np.load(root/'subbag16_rawmean_calibrated_oof.npz') as a:expected=a['probability']
 errors=[];selected=[]
 for fold in range(1,6):
  a=joblib.load(root/f'fold{fold}.joblib');base=a['base_artifact'];chosen=base['adaptive_CNN_mix'];gamma=chosen['gamma'];selected.append(chosen)
  with np.load(driver.JOB/f'cached_group_diagnostics_20261030/fold{fold}_aligned_banks.npz') as b:ix=b['outer_indices'];bank=b['outer_probability'][np.r_[np.arange(83),np.arange(104,107)]]
  extra=context_features(d,ix,a['context'])
  if gamma:
   prior=json.loads((driver.JOB/f'neural_trial_consistency_20261031/fold{fold}_selection.json').read_text())['chosen_gamma'];path=driver.JOB/f'neural_trial_consistency_20261031/selected_fold{fold}_outer.npz' if gamma==prior else root/f'gamma{gamma}_fold{fold}_outer.npz'
   with np.load(path) as b:
    valid=reverse[b['validation']]>=0;assert np.array_equal(reverse[b['validation'][valid]],ix);z=logit(np.clip(b['probability'][valid],1e-6,1-1e-6))
   q=postprocess(center(z,g[ix])/base['consistency_neural_source_scales'][0],g[ix],'conditional',.6);weight=chosen['alpha']*(np.ones(len(ix)) if chosen['gate']=='constant' else extra[:,6]);bank[83]=(1-weight)*bank[83]+weight*q
  with np.load(driver.JOB/f'aligned_covariance_svm_20261030/fold{fold}_outer.npz') as b:cov=b['logits']
  bank=np.vstack([bank,postprocess(center(cov,g[ix])/base['SPD_source_scale'],g[ix],'conditional',.6)])
  x=base['scaler'].transform(np.column_stack([features(bank,g[ix]),extra]));v=np.column_stack([x,.5*(2*extra[:,6]-1)[:,None]*x[:,[83,84,85,86]]]);q=np.mean([expit(v@t[:-1]+t[-1]) for t in a['models']['subbag16']],axis=0);mode=json.loads((driver.JOB/f'inner_calibration_20261028/fold{fold}_selection.json').read_text())['chosen'];q=calibrate(q,g[ix],mode);error=float(abs(q-expected[full['model_row_to_full_row'][ix]]).max());assert error<1e-9;errors.append(error)
 workbook=Path('/workspace/rank16_release/EEG_RANK16_20261030/submission_rank16.xlsx');sha=hashlib.sha256(workbook.read_bytes()).hexdigest();assert sha=='dc0bd4751d35badc3731533edcb8f6374c8996d050e61c1c946d5eeeb642dec4'
 driver.save(root/'independent_reload.json',dict(errors=errors,selected=selected,primary_metric='unweighted_pooled_auc',old_base_raw_MAT_replay=False,new_CNN_cached_forecasts_used=True,query_emotion_and_diagnosis_masked=True,baseline_workbook_sha256=sha,public_signal_training=False,official_submissions=0));print('RELOAD OK',max(errors),flush=True)
if __name__=='__main__':main()
