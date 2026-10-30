"""Source-inner selection of robust DE kernels, no public training or labels.
Repeated development cohort; exploratory confirmation, not a fresh test.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import json
import numpy as np
import joblib
from scipy.special import logit,expit
from scipy.stats import rankdata,norm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
import driver
from common import batches,center
from select_ensemble import postprocess

OUT=driver.JOB/'normalized_de_kernel_20261031'
OUT.mkdir(exist_ok=True)

def main():
 threadpool_limits(1)
 spec=dict(normalization=['center','zscore','rank'],gamma_multiplier=[.25,1.],C=[.3,3.],source_DEP_weight=2.,inner_objective='.75 equal-clinical pooled AUC + .25 DEP AUC',student_blend_weight=.1,conditional_tau=.6,public_training=False,outer_selection=False,source_context='eight trials of one crop, no labels',script_sha256=driver.sha(__file__))
 driver.save(OUT/'protocol.json',spec)
 with np.load(driver.ART/'features.npz') as a:d={k:a[k] for k in ['subject','trial','crop','label','diagnosis','sequence']}
 with np.load(driver.CONTROL/'full_evaluation_rows.npz') as a:full={k:a[k] for k in a.files}
 with np.load(driver.JOB/'pooled_rank_bag_20261030/subbag16_rawmean_calibrated_oof.npz') as a:teacher=a['probability']
 groups=batches(d); seq=d['sequence'].astype(float)
 # Temporal mean and variability, both derived from each10second record.
 raw=np.column_stack([seq.mean(1).reshape(len(seq),-1),np.log1p(seq.std(1)).reshape(len(seq),-1)])
 matrices={}
 for mode in spec['normalization']:
  xx=center(raw,groups,zscore=mode=='zscore')
  if mode=='rank':
   for g in np.unique(groups):
    ix=np.flatnonzero(groups==g);xx[ix]=norm.ppf((rankdata(raw[ix],axis=0)-.5)/len(ix))
  assert np.isfinite(xx).all();matrices[mode]=xx
 config=[dict(normalization=m,gamma=ga,C=c) for m in spec['normalization'] for ga in spec['gamma_multiplier'] for c in spec['C']]
 candidate=teacher.copy();student=np.full(len(teacher),.5);records=[]
 for f in driver.protocol()['folds']:
  tr=np.flatnonzero(np.isin(d['subject'],f['training_subjects']));va=np.flatnonzero(np.isin(d['subject'],f['validation_subjects']));lookup={v:j for j,v in enumerate(tr)}
  assert not set(d['subject'][tr])&set(d['subject'][va])
  inner=np.full((len(config),len(tr)),np.nan)
  for part,split in enumerate(f['inner']):
   it=np.flatnonzero(np.isin(d['subject'],split['training_subjects']));iv=np.flatnonzero(np.isin(d['subject'],split['validation_subjects']));dest=[lookup[i] for i in iv]
   assert not set(d['subject'][it])&set(d['subject'][iv])
   for mode in spec['normalization']:
    scaler=StandardScaler().fit(matrices[mode][it]);xt=scaler.transform(matrices[mode][it]);xv=scaler.transform(matrices[mode][iv])
    for ga in spec['gamma_multiplier']:
     kt=rbf_kernel(xt,gamma=ga/xt.shape[1]);kv=rbf_kernel(xv,xt,gamma=ga/xt.shape[1])
     for j,c in enumerate(config):
      if c['normalization']!=mode or c['gamma']!=ga:continue
      model=SVC(kernel='precomputed',C=c['C'],cache_size=128).fit(kt,d['label'][it],sample_weight=np.where(d['diagnosis'][it]==1,2.,1.))
      z=model.decision_function(kv);scale=max(float(np.std(center(model.decision_function(kt),groups[it]))),.1)
      inner[j,dest]=postprocess(center(z,groups[iv])/scale,groups[iv],'conditional',.6)
   print('DE KERNEL',f['fold'],'inner',part+1,'/3 complete',flush=True)
  assert np.isfinite(inner).all()
  dep=d['diagnosis'][tr]==1;y=d['label'][tr]
  metrics=[dict(config=c,pooled_balanced=float(roc_auc_score(y,p,sample_weight=np.where(dep,2.,1.))),DEP=float(roc_auc_score(y[dep],p[dep]))) for c,p in zip(config,inner)]
  chosen=int(np.argmax([.75*m['pooled_balanced']+.25*m['DEP'] for m in metrics]));c=config[chosen];mode=c['normalization']
  scaler=StandardScaler().fit(matrices[mode][tr]);xt=scaler.transform(matrices[mode][tr]);xv=scaler.transform(matrices[mode][va]);gamma=c['gamma']/xt.shape[1];kt=rbf_kernel(xt,gamma=gamma);kv=rbf_kernel(xv,xt,gamma=gamma)
  model=SVC(kernel='precomputed',C=c['C'],cache_size=128).fit(kt,y,sample_weight=np.where(dep,2.,1.));scale=max(float(np.std(center(model.decision_function(kt),groups[tr]))),.1)
  q=postprocess(center(model.decision_function(kv),groups[va])/scale,groups[va],'conditional',.6);dest=full['model_row_to_full_row'][va];student[dest]=q
  candidate[dest]=expit(.9*logit(np.clip(teacher[dest],1e-6,1-1e-6))+.1*logit(np.clip(q,1e-6,1-1e-6)))
  joblib.dump(dict(model=model,scaler=scaler,reference=xt,scale=scale,config=c,source_subjects=f['training_subjects']),OUT/f'fold{f["fold"]}.joblib',compress=3)
  np.savez_compressed(OUT/f'fold{f["fold"]}_inner.npz',indices=tr,probability=inner)
  records.append(dict(fold=f['fold'],chosen=c,inner_metrics=metrics));print('DE KERNEL OUTER',f['fold'],'/5 complete',flush=True)
 # All outer fitting and selections finished before outer labels are scored.
 y=full['label'];dep=full['diagnosis']==1;results={}
 for name,p in [('frozen_teacher',teacher),('selected_student',student),('fixed_10percent_student_blend',candidate)]:
  results[name]=dict(pooled_auc=float(roc_auc_score(y,p)),equal_clinical_mass_pooled_auc=float(roc_auc_score(y,p,sample_weight=np.where(dep,2.,1.))),DEP_auc=float(roc_auc_score(y[dep],p[dep])))
 np.savez_compressed(OUT/'development_predictions.npz',teacher=teacher,student=student,candidate=candidate)
 driver.save(OUT/'summary.json',dict(results=results,folds=records,outer_cohort_reused=True,fresh_independent_confirmation=False,baseline_not_replaced=True,public_signals_used=False,official_submission=False))
 print('ALL COMPLETE',json.dumps(results),flush=True)
if __name__=='__main__':main()
