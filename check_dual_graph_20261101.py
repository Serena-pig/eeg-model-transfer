"""Independent selected temporal/static expert and fusion reload on cached EEG.
No old83 raw-MAT replay or official test inference.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import json,hashlib
from pathlib import Path
import numpy as np,torch,joblib
from scipy.special import logit,expit
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
import driver
from connectivity_graph_20261101 import Data,Graph,infer as infer_static
from temporal_graph_20261101 import TaskData,TemporalGraph,infer
from leaderboard_dualbranch_20261028 import groups
from common import batches,center
from cached_fixed_fusion_20261030 import features
from context_rank_v34 import context_features
from select_ensemble import postprocess
from inner_calibration_round import calibrate

def main():
 threadpool_limits(1);torch.set_num_threads(1);data=Data();full=data.d;full['label'][:]=-1;full['diagnosis'][:]=-1;root=driver.JOB/'temporal_graph_20261101';fusion=driver.JOB/'dual_graph_fusion_20261101'
 with np.load(driver.ART/'features.npz') as a:d={k:a[k] for k in ['subject','trial','crop','label','diagnosis','train','powers','quality']}
 d['label'][:]=-1;d['diagnosis'][:]=-1;g=batches(d);reverse=np.full(len(full['label']),-1,dtype=int);reverse[full['model_row_to_full_row']]=np.arange(len(d['subject']));errors=[];headerrors=[];selected=np.full(len(full['label']),.5)
 for f in driver.protocol()['folds']:
  k=f['fold'];chosen=json.loads((root/f'fold{k}_selection.json').read_text())['chosen'];va=np.flatnonzero(np.isin(full['subject'],f['validation_subjects']));u,rr=groups(full,va);parent=driver.JOB/f'connectivity_graph_20261101/selected_fold{k}_outer.pt'
  if chosen=='static':checkpoint=torch.load(parent,map_location='cpu',weights_only=True);model=Graph();task=data;infer_fn=infer_static
  else:
   tr=np.flatnonzero(np.isin(full['subject'],f['training_subjects'])&data.eligible);task=TaskData(data,tr,va,parent);checkpoint=torch.load(root/f'selected_fold{k}_outer.pt',map_location='cpu',weights_only=True);assert checkpoint['parent_graph_sha256']==driver.sha(parent);model=TemporalGraph(chosen);infer_fn=infer
  zz=[]
  for state in checkpoint['states']:model.load_state_dict(state);zz.append(infer_fn(model,task,rr))
  q=postprocess(np.mean(zz,0).ravel(),np.repeat(u,8),'conditional',1.);qa=np.zeros(len(va));lookup={v:j for j,v in enumerate(va)};qa[[lookup[v] for v in rr.ravel()]]=q
  with np.load(root/f'selected_fold{k}_outer.npz') as a:assert np.array_equal(a['validation'],va);error=float(abs(qa-a['probability']).max());assert error<1e-6;errors.append(error)
  selected[va]=qa;artifact=joblib.load(fusion/f'fold{k}.joblib');base=artifact['base_artifact']
  with np.load(driver.JOB/f'cached_group_diagnostics_20261030/fold{k}_aligned_banks.npz') as a:ix=a['outer_indices'];bank=a['outer_probability'][np.r_[np.arange(83),np.arange(104,107)]]
  valid=reverse[va]>=0;assert np.array_equal(reverse[va[valid]],ix);newrow=postprocess(center(logit(np.clip(qa[valid],1e-6,1-1e-6)),g[ix])/base['consistency_neural_source_scales'][0],g[ix],'conditional',.6)
  with np.load(driver.JOB/f'aligned_covariance_svm_20261030/fold{k}_outer.npz') as a:cov=a['logits']
  with np.load(driver.JOB/f'connectivity_graph_20261101/selected_fold{k}_outer.npz') as cached:
   assert np.array_equal(cached['validation'],va);rawq=cached['probability'][valid]
  rawrow=postprocess(center(logit(np.clip(rawq,1e-6,1-1e-6)),g[ix])/base['static_graph_source_scale'],g[ix],'conditional',.6)
  bank=np.vstack([bank,postprocess(center(cov,g[ix])/base['SPD_source_scale'],g[ix],'conditional',.6),rawrow,newrow]);extra=context_features(d,ix,artifact['context']);x=base['scaler'].transform(np.column_stack([features(bank,g[ix]),extra]));v=np.column_stack([x,.5*(2*extra[:,6]-1)[:,None]*x[:,[83,84,85,86,87,88]]]);qf=np.mean([expit(v@t[:-1]+t[-1]) for t in artifact['models']['subbag16']],0);mode=json.loads((driver.JOB/f'inner_calibration_20261028/fold{k}_selection.json').read_text())['chosen'];qf=calibrate(qf,g[ix],mode)
  with np.load(fusion/'subbag16_rawmean_calibrated_oof.npz') as a:expected=a['probability'][full['model_row_to_full_row'][ix]]
  error=float(abs(qf-expected).max());assert error<1e-6;headerrors.append(error)
 workbook=Path('/workspace/rank16_release/EEG_RANK16_20261030/submission_rank16.xlsx');sha=hashlib.sha256(workbook.read_bytes()).hexdigest();assert sha=='dc0bd4751d35badc3731533edcb8f6374c8996d050e61c1c946d5eeeb642dec4'
 result=dict(expert_reload_errors=errors,fusion_reload_errors=headerrors,query_labels_and_diagnoses_masked=True,old83_raw_MAT_replayed=False,baseline_workbook_sha256=sha,public_data_used=False,holdout_used=False,official_submissions=0);driver.save(fusion/'independent_reload.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
