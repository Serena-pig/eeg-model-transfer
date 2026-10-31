"""Nonlinear10s student with source-only50s-trial prediction consistency.
Exact source3inner subject selection; no public inputs, no target50s inference.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json
import numpy as np,joblib
from scipy.linalg import cho_factor,cho_solve
from scipy.special import expit,logit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
import driver
from common import batches,center
from select_ensemble import postprocess
OUT=driver.JOB/'trial_consistency_20261031';OUT.mkdir(exist_ok=True)
DIM=512;LAMBDAS=[.3,3.,30.];GAMMAS=[0.,.1,1.,10.]

def setup(raw,tr):
 scaler=StandardScaler().fit(raw[tr]);x=scaler.transform(raw[tr]);rng=np.random.default_rng(1031);projection=rng.normal(size=(x.shape[1],DIM))*np.sqrt(2/x.shape[1]);bias=rng.uniform(0,2*np.pi,DIM)
 artifact=dict(scaler=scaler,projection=projection,bias=bias)
 return artifact,transform(raw[tr],artifact)

def transform(x,a):return np.sqrt(2/DIM)*np.cos(a['scaler'].transform(x)@a['projection']+a['bias'])

def fit_all(x,y,trial_groups):
 # Supervised source only. Each group is one real50s trial, exactly5crops.
 dev=x.copy()
 for g in np.unique(trial_groups):
  mask=trial_groups==g;assert mask.sum()==5;assert len(np.unique(y[mask]))==1
  dev[mask]-=x[mask].mean(0)
 gram=x.T@x;within=dev.T@dev;target=x.T@(y-.5);models={}
 for gamma in GAMMAS:
  for lam in LAMBDAS:
   mat=gram+gamma*within+lam*np.eye(DIM);coef=cho_solve(cho_factor(mat,lower=True,check_finite=False),target,check_finite=False);models[(gamma,lam)]=coef
 return models

def predict(x,coef,source_x,groups):
 scale=max(float(np.std(source_x@coef)),.05)
 return postprocess(center(x@coef,groups)/scale,groups,'conditional',.6)

def main():
 threadpool_limits(1);spec=dict(model='RBF random Fourier feature ridge student',features=DIM,seed=1031,ridge=LAMBDAS,consistency=GAMMAS,loss='sum((f-y)^2)+gamma*sum((f-mean_same_source_trial(f))^2)+lambda*norm(coef)^2',inner_metric='mean equal-clinical-mass pooled AUC across3source inner folds',source_groups='subject+trial,5different10second source crops',prediction_context='8trials of same person and crop, never other50s crops',deployment_input_seconds=10,fixed_blend_student_weight=.1,control='same learner, gamma0 selected lambda using same3inner folds',public_training=False,script_sha256=driver.sha(__file__))
 driver.save(OUT/'protocol.json',spec)
 with np.load(driver.ART/'features.npz') as a:d={k:a[k] for k in ['subject','trial','crop','label','diagnosis','sequence','relative']}
 with np.load(driver.CONTROL/'full_evaluation_rows.npz') as a:full={k:a[k] for k in a.files}
 with np.load(driver.JOB/'pooled_rank_bag_20261030/subbag16_rawmean_calibrated_oof.npz') as a:teacher=a['probability']
 g=batches(d);source_trials=np.char.add(np.char.add(d['subject'],':trial'),d['trial'].astype(str));s=d['sequence'];raw=np.column_stack([s.mean(1).reshape(len(g),-1),np.log1p(s.std(1)).reshape(len(g),-1),d['relative'].reshape(len(g),-1)]);raw=center(raw,g,zscore=True)
 config=[(gamma,lam) for gamma in GAMMAS for lam in LAMBDAS];outputs={name:teacher.copy() for name in ['control_student','consistency_student','control_blend','consistency_blend']};selections=[];reloads=[]
 for f in driver.protocol()['folds']:
  metrics={c:[] for c in config}
  for j,sp in enumerate(f['inner']):
   tr=np.flatnonzero(np.isin(d['subject'],sp['training_subjects']));va=np.flatnonzero(np.isin(d['subject'],sp['validation_subjects']));assert not set(d['subject'][tr])&set(d['subject'][va]);a,x=setup(raw,tr);v=transform(raw[va],a);models=fit_all(x,d['label'][tr],source_trials[tr]);dep=d['diagnosis'][va]==1
   predictions=[]
   for c in config:
    q=predict(v,models[c],x,g[va]);metrics[c].append(float(roc_auc_score(d['label'][va],q,sample_weight=np.where(dep,2.,1.))));predictions.append(q)
   np.savez_compressed(OUT/f'fold{f["fold"]}_inner{j}.npz',indices=va,probabilities=np.stack(predictions))
   print('CONSISTENCY INNER',f['fold'],j+1,'/3 COMPLETE',flush=True)
  best=max(config,key=lambda c:np.mean(metrics[c]));control=max([c for c in config if c[0]==0],key=lambda c:np.mean(metrics[c]));assert best in config
  tr=np.flatnonzero(np.isin(d['subject'],f['training_subjects']));va=np.flatnonzero(np.isin(d['subject'],f['validation_subjects']));a,x=setup(raw,tr);v=transform(raw[va],a);models=fit_all(x,d['label'][tr],source_trials[tr]);artifact=dict(transform=a,models={'control':models[control],'consistency':models[best]},scale={name:max(float(np.std(x@models[c])),.05) for name,c in [('control',control),('consistency',best)]},selected={'control':control,'consistency':best},source_subjects=f['training_subjects'],protocol=spec);path=OUT/f'fold{f["fold"]}.joblib';joblib.dump(artifact,path,compress=3);saved=joblib.load(path);vr=transform(raw[va],saved['transform']);dest=full['model_row_to_full_row'][va]
  for name,c in [('control',control),('consistency',best)]:
   q=predict(v,models[c],x,g[va]);replay=postprocess(center(vr@saved['models'][name],g[va])/saved['scale'][name],g[va],'conditional',.6);error=float(abs(q-replay).max());assert error<1e-10;reloads.append(error);outputs[name+'_student'][dest]=q;outputs[name+'_blend'][dest]=expit(.9*logit(np.clip(teacher[dest],1e-6,1-1e-6))+.1*logit(np.clip(q,1e-6,1-1e-6)))
  selection=dict(fold=f['fold'],chosen=best,control=control,inner=[dict(gamma=c[0],ridge=c[1],fold_scores=metrics[c],mean=float(np.mean(metrics[c]))) for c in config]);selections.append(selection);driver.save(OUT/f'fold{f["fold"]}_selection.json',selection);print('CONSISTENCY OUTER',f['fold'],'/5 COMPLETE chosen',best,flush=True)
 # All selections and fitting precede any outer scoring.
 y=full['label'];dep=full['diagnosis']==1;outputs['teacher']=teacher;results={}
 for name,q in outputs.items():
  results[name]=dict(pooled_auc=float(roc_auc_score(y,q)),equal_clinical_mass_pooled_auc=float(roc_auc_score(y,q,sample_weight=np.where(dep,2.,1.))),DEP_auc=float(roc_auc_score(y[dep],q[dep])));np.savez_compressed(OUT/(name+'_oof.npz'),probability=q,subjects=full['subject'])
 driver.save(OUT/'summary.json',dict(results=results,selections=selections,max_saved_model_reload_error=max(reloads),all5outer3inner_complete=True,source_only=True,same_trial_split_never_crosses_people=True,public_signals_read=False,official_submission=False,baseline_preserved=True,official_baseline_user_report=88.13,outer_cohort_reused=True,independent_confirmation=False))
 print('ALL COMPLETE',json.dumps(results),flush=True)
if __name__=='__main__':main()
