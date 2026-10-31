"""Independent reload of new fusion heads on cached held-subject base forecasts.
This is head replay, not a new raw-EEG replay of every base member.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import sys,hashlib
from pathlib import Path
import numpy as np,joblib
from scipy.special import expit
from threadpoolctl import threadpool_limits
import driver
from common import batches,center
from cached_fixed_fusion_20261030 import features
from context_rank_v34 import context_features
from select_ensemble import postprocess
from inner_calibration_round import calibrate

def main():
 threadpool_limits(1)
 with np.load(driver.ART/'features.npz') as a:d={k:a[k] for k in ['subject','trial','crop','label','diagnosis','train','powers','quality']}
 d['label'][:]=-1;d['diagnosis'][:]=-1
 with np.load(driver.CONTROL/'full_evaluation_rows.npz') as a:mapping=a['model_row_to_full_row']
 groups=batches(d);baseix=np.r_[np.arange(83),np.arange(104,107)]
 for name in sys.argv[1:]:
  root=driver.JOB/name
  with np.load(root/'subbag16_rawmean_calibrated_oof.npz') as a:expected=a['probability']
  errors=[];hashes={}
  for fold in range(1,6):
   p=root/f'fold{fold}.joblib';a=joblib.load(p);hashes[p.name]=driver.sha(p)
   with np.load(driver.JOB/f'cached_group_diagnostics_20261030/fold{fold}_aligned_banks.npz') as b:va=b['outer_indices'];bank=b['outer_probability'][baseix]
   with np.load(driver.JOB/f'aligned_covariance_svm_20261030/fold{fold}_outer.npz') as b:assert np.array_equal(va,b['validation']);z=b['logits']
   base=a['base_artifact'];bank=np.vstack([bank,postprocess(center(z,groups[va])/base['SPD_source_scale'],groups[va],'conditional',.6)])
   e=context_features(d,va,a['context']);x=base['scaler'].transform(np.column_stack([features(bank,groups[va]),e]));v=np.column_stack([x,.5*(2*e[:,6]-1)[:,None]*x[:,[83,84,85,86]]])
   if 'columns' in a:v=v[:,a['columns']]
   weights=a['models']['subbag16'];q=np.mean([expit(v@t[:-1]+t[-1]) for t in weights],axis=0)
   mode=__import__('json').loads((driver.JOB/f'inner_calibration_20261028/fold{fold}_selection.json').read_text())['chosen'];q=calibrate(q,groups[va],mode)
   err=float(abs(q-expected[mapping[va]]).max());assert err<1e-9;errors.append(err)
  workbook=Path('/workspace/rank16_release/EEG_RANK16_20261030/submission_rank16.xlsx');sha=hashlib.sha256(workbook.read_bytes()).hexdigest();assert sha=='dc0bd4751d35badc3731533edcb8f6374c8996d050e61c1c946d5eeeb642dec4'
  result=dict(check='independent native fusion-head reload on cached outer base forecasts',full_raw_base_replay=False,errors=errors,head_sha256=hashes,baseline_workbook_unchanged=True,official_feedback_baseline=88.13,public_inputs_not_read=True,submissions_performed=False)
  driver.save(root/'independent_head_replay.json',result);print(name,max(errors),flush=True)
if __name__=='__main__':main()
