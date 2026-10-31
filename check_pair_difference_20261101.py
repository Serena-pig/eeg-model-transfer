"""Native comparator plus89member fusion replay; labels/diagnoses masked.
Uses cached old graph/base forecasts; no old raw-MAT replay.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import sys,json,importlib,hashlib
from pathlib import Path
import numpy as np,joblib
from scipy.special import logit,expit
from threadpoolctl import threadpool_limits
import driver
from common import batches,center
from context_rank_v34 import context_features
from cached_fixed_fusion_20261030 import features
from select_ensemble import postprocess
from inner_calibration_round import calibrate

def main():
 threadpool_limits(1)
 for name in sys.argv[1:]:
  module=importlib.import_module(name);source=driver.JOB/name;fusion=driver.JOB/(name.replace('pair_difference','pair_difference_fusion').replace('pair_linear_range','pair_linear_range_fusion'));data=module.Data();full=data.full;data.d['label'][:]=-1;data.d['diagnosis'][:]=-1;full['label'][:]=-1;full['diagnosis'][:]=-1
  with np.load(driver.ART/'features.npz') as a:d={k:a[k] for k in ['subject','trial','crop','label','diagnosis','train','powers','quality']}
  d['label'][:]=-1;d['diagnosis'][:]=-1;g=batches(d);errors=[];headerrors=[]
  for f in driver.protocol()['folds']:
   k=f['fold'];va=np.flatnonzero(np.isin(full['subject'],f['validation_subjects']));chosen=json.loads((source/f'fold{k}_selection.json').read_text())['chosen'];assert chosen!='old_static';comparator=joblib.load(source/f'fold{k}.joblib');q=module.predict(data,va,comparator)
   with np.load(source/f'selected_fold{k}_outer.npz') as a:assert np.array_equal(a['validation'],va);error=float(abs(q-a['probability']).max());assert error<1e-12;errors.append(error)
   artifact=joblib.load(fusion/f'fold{k}.joblib');base=artifact['base_artifact']
   with np.load(driver.JOB/f'cached_group_diagnostics_20261030/fold{k}_aligned_banks.npz') as a:ix=a['outer_indices'];bank=a['outer_probability'][np.r_[np.arange(83),np.arange(104,107)]]
   valid=data.reverse[va]>=0;assert np.array_equal(data.reverse[va[valid]],ix);newrow=postprocess(center(logit(np.clip(q[valid],1e-6,1-1e-6)),g[ix])/base['consistency_neural_source_scales'][0],g[ix],'conditional',.6)
   with np.load(driver.JOB/f'connectivity_graph_20261101/selected_fold{k}_outer.npz') as a:rawq=a['probability'][valid]
   rawrow=postprocess(center(logit(np.clip(rawq,1e-6,1-1e-6)),g[ix])/base['static_graph_source_scale'],g[ix],'conditional',.6)
   with np.load(driver.JOB/f'aligned_covariance_svm_20261030/fold{k}_outer.npz') as a:cov=postprocess(center(a['logits'],g[ix])/base['SPD_source_scale'],g[ix],'conditional',.6)
   bank=np.vstack([bank,cov,rawrow,newrow]);ex=context_features(d,ix,artifact['context']);x=base['scaler'].transform(np.column_stack([features(bank,g[ix]),ex]));v=np.column_stack([x,.5*(2*ex[:,6]-1)[:,None]*x[:,[83,84,85,86,87,88]]]);prob=np.mean([expit(v@t[:-1]+t[-1]) for t in artifact['models']['subbag16']],0);mode=json.loads((driver.JOB/f'inner_calibration_20261028/fold{k}_selection.json').read_text())['chosen'];prob=calibrate(prob,g[ix],mode)
   with np.load(fusion/'subbag16_rawmean_calibrated_oof.npz') as a:expected=a['probability'][full['model_row_to_full_row'][ix]]
   error=float(abs(prob-expected).max());assert error<1e-9;headerrors.append(error)
  book=Path('/workspace/rank16_release/EEG_RANK16_20261030/submission_rank16.xlsx');sha=hashlib.sha256(book.read_bytes()).hexdigest();assert sha=='dc0bd4751d35badc3731533edcb8f6374c8996d050e61c1c946d5eeeb642dec4';result=dict(comparator_reload_errors=errors,fusion_reload_errors=headerrors,query_labels_and_diagnoses_masked=True,old_raw_MAT_models_replayed=False,baseline_workbook_sha256=sha,public_training=False,official_submissions=0);driver.save(fusion/'independent_reload.json',result);print(name,'RELOAD PASSED',max(headerrors),flush=True)
if __name__=='__main__':main()
