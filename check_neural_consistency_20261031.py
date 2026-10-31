"""Reload selected new CNNs on cached10s EEG features and refitted fusion.
Does not replay original83 base models from MAT or access official signals.
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
from leaderboard_dualbranch_20261028 import Net,Inputs,groups,infer
from common import batches,center
from cached_fixed_fusion_20261030 import features
from context_rank_v34 import context_features
from select_ensemble import postprocess
from inner_calibration_round import calibrate

def main():
 threadpool_limits(1);torch.set_num_threads(1);inputs=Inputs();full=inputs.d;root=driver.JOB/'neural_trial_consistency_20261031';fusion=driver.JOB/'neural_consistency_fusion_20261031'
 with np.load(driver.ART/'features.npz') as a:d={k:a[k] for k in ['subject','trial','crop','label','diagnosis','train','powers','quality']}
 d['label'][:]=-1;d['diagnosis'][:]=-1;g=batches(d);reverse=inputs.reverse;selected=np.full(len(full['label']),.5);baseline=selected.copy();errors=[];headerrors=[];baseix=np.r_[np.arange(83),np.arange(104,107)]
 for f in driver.protocol()['folds']:
  k=f['fold'];sel=json.loads((root/f'fold{k}_selection.json').read_text());gamma=sel['chosen_gamma'];path=root/f'selected_fold{k}_outer.pt' if gamma else driver.JOB/f'leaderboard_dualbranch_20261028/de_cnn_fold{k}_outer.pt';a=torch.load(path,map_location='cpu',weights_only=True)
  va=np.flatnonzero(np.isin(full['subject'],f['validation_subjects']));u,rr=groups(full,va);model=Net('de_cnn');zz=[]
  for state in a['states']:model.load_state_dict(state);zz.append(infer(model,inputs,rr))
  q=postprocess(np.mean(zz,axis=0).ravel(),np.repeat(u,8),'conditional',1.);qa=np.zeros(len(va));lookup={v:j for j,v in enumerate(va)};qa[[lookup[v] for v in rr.ravel()]]=q
  with np.load(root/f'selected_fold{k}_outer.npz') as b:assert np.array_equal(va,b['validation']);error=float(abs(qa-b['probability']).max());assert error<1e-6;errors.append(error)
  selected[va]=qa
  with np.load(driver.JOB/f'leaderboard_dualbranch_20261028/de_cnn_fold{k}_outer.npz') as b:baseline[va]=b['probability']
  artifact=joblib.load(fusion/f'fold{k}.joblib');base=artifact['base_artifact']
  with np.load(driver.JOB/f'cached_group_diagnostics_20261030/fold{k}_aligned_banks.npz') as b:ix=b['outer_indices'];bank=b['outer_probability'][baseix]
  valid=reverse[va]>=0;assert np.array_equal(reverse[va[valid]],ix);z=logit(np.clip(qa[valid],1e-6,1-1e-6));bank[83]=postprocess(center(z,g[ix])/base['consistency_neural_source_scales'][0],g[ix],'conditional',.6)
  with np.load(driver.JOB/f'aligned_covariance_svm_20261030/fold{k}_outer.npz') as b:cov=b['logits']
  bank=np.vstack([bank,postprocess(center(cov,g[ix])/base['SPD_source_scale'],g[ix],'conditional',.6)]);ex=context_features(d,ix,artifact['context']);x=base['scaler'].transform(np.column_stack([features(bank,g[ix]),ex]));v=np.column_stack([x,.5*(2*ex[:,6]-1)[:,None]*x[:,[83,84,85,86]]]);thetas=artifact['models']['subbag16'];raw=np.mean([expit(v@t[:-1]+t[-1]) for t in thetas],axis=0);mode=json.loads((driver.JOB/f'inner_calibration_20261028/fold{k}_selection.json').read_text())['chosen'];qf=calibrate(raw,g[ix],mode)
  with np.load(fusion/'subbag16_rawmean_calibrated_oof.npz') as b:expected=b['probability'][full['model_row_to_full_row'][ix]]
  delta=float(abs(qf-expected).max());assert delta<1e-6;headerrors.append(delta)
 y=full['label'];dep=full['diagnosis']==1;metrics={}
 for name,q in [('old_CNN',baseline),('selected_consistency_CNN',selected)]:metrics[name]=dict(pooled_auc=float(roc_auc_score(y,q)),equal_clinical_mass_pooled_auc=float(roc_auc_score(y,q,sample_weight=np.where(dep,2.,1.))),DEP_auc=float(roc_auc_score(y[dep],q[dep])))
 workbook=Path('/workspace/rank16_release/EEG_RANK16_20261030/submission_rank16.xlsx');sha=hashlib.sha256(workbook.read_bytes()).hexdigest();assert sha=='dc0bd4751d35badc3731533edcb8f6374c8996d050e61c1c946d5eeeb642dec4'
 result=dict(cached_EEG_feature_native_CNN_errors=errors,native_fusion_errors=headerrors,old83_raw_MAT_replayed=False,standalone_CNN_metrics=metrics,baseline_workbook_sha256=sha,public_signals_used=False,holdout_used=False,official_submissions=0)
 driver.save(fusion/'independent_reload.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
