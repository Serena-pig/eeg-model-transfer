"""Reload selected local graph plus matching88/89member fusion on cached EEG.
No public inputs or old83 raw-MAT inference.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import json,hashlib
from pathlib import Path
import numpy as np,torch,joblib
from scipy.special import logit,expit
from threadpoolctl import threadpool_limits
import driver
from dynamic_connectivity_20261101 import Data,DynamicGraph,infer
from connectivity_graph_20261101 import Data as OldData,Graph,infer as old_infer
from leaderboard_dualbranch_20261028 import groups
from common import batches,center
from cached_fixed_fusion_20261030 import features
from context_rank_v34 import context_features
from select_ensemble import postprocess
from inner_calibration_round import calibrate

def main():
 threadpool_limits(1);torch.set_num_threads(1);data=Data();full=data.d;full['label'][:]=-1;full['diagnosis'][:]=-1;root=driver.JOB/'dynamic_connectivity_20261101';fusion=driver.JOB/'dynamic_connectivity_fusion_20261101'
 with np.load(driver.ART/'features.npz') as a:d={k:a[k] for k in ['subject','trial','crop','label','diagnosis','train','powers','quality']}
 d['label'][:]=-1;d['diagnosis'][:]=-1;g=batches(d);reverse=data.reverse;errors=[];headerrors=[]
 for f in driver.protocol()['folds']:
  k=f['fold'];chosen=json.loads((root/f'fold{k}_selection.json').read_text())['chosen'];va=np.flatnonzero(np.isin(full['subject'],f['validation_subjects']));u,rr=groups(full,va)
  if chosen=='old_static':
   model=Graph();task=OldData();task.d['label'][:]=-1;task.d['diagnosis'][:]=-1;checkpoint=torch.load(driver.JOB/f'connectivity_graph_20261101/selected_fold{k}_outer.pt',map_location='cpu',weights_only=True);run=lambda:old_infer(model,task,rr)
  else:
   model=DynamicGraph();checkpoint=torch.load(root/f'selected_fold{k}_outer.pt',map_location='cpu',weights_only=True);assert checkpoint['mode']==chosen;run=lambda:infer(model,data,rr,chosen)
  zz=[]
  for state in checkpoint['states']:model.load_state_dict(state);zz.append(run())
  qa=np.zeros(len(va));lookup={v:j for j,v in enumerate(va)};qa[[lookup[v] for v in rr.ravel()]]=postprocess(np.mean(zz,0).ravel(),np.repeat(u,8),'conditional',1.)
  with np.load(root/f'selected_fold{k}_outer.npz') as a:assert np.array_equal(a['validation'],va);error=float(abs(qa-a['probability']).max());assert error<1e-6;errors.append(error)
  artifact=joblib.load(fusion/f'fold{k}.joblib');base=artifact['base_artifact']
  with np.load(driver.JOB/f'cached_group_diagnostics_20261030/fold{k}_aligned_banks.npz') as a:ix=a['outer_indices'];bank=a['outer_probability'][np.r_[np.arange(83),np.arange(104,107)]]
  valid=reverse[va]>=0;assert np.array_equal(reverse[va[valid]],ix);newrow=postprocess(center(logit(np.clip(qa[valid],1e-6,1-1e-6)),g[ix])/base['consistency_neural_source_scales'][0],g[ix],'conditional',.6)
  with np.load(driver.JOB/f'aligned_covariance_svm_20261030/fold{k}_outer.npz') as a:cov=postprocess(center(a['logits'],g[ix])/base['SPD_source_scale'],g[ix],'conditional',.6)
  if chosen=='old_static':bank=np.vstack([bank,cov,newrow]);interactions=[83,84,85,86,87]
  else:
   with np.load(driver.JOB/f'connectivity_graph_20261101/selected_fold{k}_outer.npz') as a:assert np.array_equal(a['validation'],va);rawq=a['probability'][valid]
   rawrow=postprocess(center(logit(np.clip(rawq,1e-6,1-1e-6)),g[ix])/base['static_graph_source_scale'],g[ix],'conditional',.6);bank=np.vstack([bank,cov,rawrow,newrow]);interactions=[83,84,85,86,87,88]
  extra=context_features(d,ix,artifact['context']);x=base['scaler'].transform(np.column_stack([features(bank,g[ix]),extra]));v=np.column_stack([x,.5*(2*extra[:,6]-1)[:,None]*x[:,interactions]]);assert v.shape[1]+1==artifact['models']['subbag16'].shape[1];q=np.mean([expit(v@t[:-1]+t[-1]) for t in artifact['models']['subbag16']],0);mode=json.loads((driver.JOB/f'inner_calibration_20261028/fold{k}_selection.json').read_text())['chosen'];q=calibrate(q,g[ix],mode)
  with np.load(fusion/'subbag16_rawmean_calibrated_oof.npz') as a:expected=a['probability'][full['model_row_to_full_row'][ix]]
  error=float(abs(q-expected).max());assert error<1e-6;headerrors.append(error)
 book=Path('/workspace/rank16_release/EEG_RANK16_20261030/submission_rank16.xlsx');sha=hashlib.sha256(book.read_bytes()).hexdigest();assert sha=='dc0bd4751d35badc3731533edcb8f6374c8996d050e61c1c946d5eeeb642dec4'
 result=dict(selected_network_reload_errors=errors,fusion_reload_errors=headerrors,query_emotion_and_diagnosis_masked=True,old83_raw_MAT_replayed=False,baseline_workbook_sha256=sha,public_data_used=False,holdout_used=False,official_submissions=0);driver.save(fusion/'independent_reload.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
