"""Fixed two-parameter residual on frozen rank16, no full-meta refitting.
Fitted source meta forecasts are training data, not claimed nested validation.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import json
import numpy as np,joblib
from scipy.special import expit,logit
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
import driver
from common import batches,center
from cached_fixed_fusion_20261030 import features
from context_rank_v34 import fit_context,context_features
from select_ensemble import postprocess
from inner_calibration_round import calibrate
from postflight import pair_matrix
OUT=driver.JOB/'seed_stability_20261101/seed1035/residual';OUT.mkdir(exist_ok=True)

def main():
 threadpool_limits(1);spec=dict(primary_metric='unweighted_pooled_auc',fixed_C=.01,correction_parameters=2,features='centered new-old CNN logit; clinical CF interaction; both gated by frozen teacher uncertainty',teacher_frozen=True,loss='.5 BCE + .5 exact pooled pair logistic + ridge',hyperparameters_not_selected_on_outer=True,source_meta_in_sample_used_for_fit_not_validation=True,public_training=False,script_sha256=driver.sha(__file__))
 driver.save(OUT/'protocol.json',spec)
 with np.load(driver.ART/'features.npz') as a:d={k:a[k] for k in ['subject','trial','crop','label','diagnosis','train','powers','quality']}
 with np.load(driver.CONTROL/'full_evaluation_rows.npz') as a:full={k:a[k] for k in a.files}
 with np.load(driver.JOB/'seed_stability_20261101/seed1035/baseline/subbag16_rawmean_calibrated_oof.npz') as a:anchor=a['probability']
 result=anchor.copy();gg=batches(d);reverse=np.full(len(full['label']),-1,dtype=int);reverse[full['model_row_to_full_row']]=np.arange(len(d['label']));records=[]
 for f in driver.protocol()['folds']:
  k=f['fold']
  with np.load(driver.JOB/f'cached_group_diagnostics_20261030/fold{k}_aligned_banks.npz') as a:tr=a['inner_indices'];va=a['outer_indices'];ib=a['inner_probability'][np.r_[np.arange(83),np.arange(104,107)]];ob=a['outer_probability'][np.r_[np.arange(83),np.arange(104,107)]]
  lookup={v:j for j,v in enumerate(tr)};zi=np.zeros(len(tr));newzi=np.zeros(len(tr))
  for part in range(3):
   with np.load(driver.JOB/f'aligned_covariance_svm_20261030/fold{k}_inner{part}.npz') as a:zi[[lookup[v] for v in a['validation']]]=a['logits']
   with np.load(driver.JOB/f'neural_trial_consistency_20261031/selected_fold{k}_inner{part}.npz') as a:
    good=reverse[a['validation']]>=0;rows=reverse[a['validation'][good]];newzi[[lookup[v] for v in rows]]=logit(np.clip(a['probability'][good],1e-6,1-1e-6))
  with np.load(driver.JOB/f'aligned_covariance_svm_20261030/fold{k}_outer.npz') as a:zv=a['logits']
  with np.load(driver.JOB/f'neural_trial_consistency_20261031/selected_fold{k}_outer.npz') as a:good=reverse[a['validation']]>=0;assert np.array_equal(reverse[a['validation'][good]],va);newzv=logit(np.clip(a['probability'][good],1e-6,1-1e-6))
  frozen=joblib.load(driver.JOB/f'seed_stability_20261101/seed1035/baseline/fold{k}.joblib');base=frozen['base_artifact'];gi=gg[tr];gv=gg[va];sd=base['SPD_source_scale'];ib=np.vstack([ib,postprocess(center(zi,gi)/sd,gi,'conditional',.6)]);ob=np.vstack([ob,postprocess(center(zv,gv)/sd,gv,'conditional',.6)])
  source=dict(d);source['label']=np.full_like(d['label'],-1);source['diagnosis']=np.full_like(d['diagnosis'],-1);source['diagnosis'][tr]=d['diagnosis'][tr];_,e=fit_context(source,tr);blind=dict(d);blind['label']=np.full_like(d['label'],-1);blind['diagnosis']=np.full_like(d['diagnosis'],-1);ev=context_features(blind,va,frozen['context']);x=base['scaler'].transform(np.column_stack([features(ib,gi),e]));v=base['scaler'].transform(np.column_stack([features(ob,gv),ev]));x=np.column_stack([x,.5*(2*e[:,6]-1)[:,None]*x[:,[83,84,85,86]]]);v=np.column_stack([v,.5*(2*ev[:,6]-1)[:,None]*v[:,[83,84,85,86]]]);theta=frozen['models']['subbag16'];mode=json.loads((driver.JOB/f'inner_calibration_20261028/fold{k}_selection.json').read_text())['chosen'];qi=calibrate(np.mean([expit(x@t[:-1]+t[-1]) for t in theta],0),gi,mode);qv=calibrate(np.mean([expit(v@t[:-1]+t[-1]) for t in theta],0),gv,mode);assert abs(qv-anchor[full['model_row_to_full_row'][va]]).max()<1e-9
  scale=max(float(center(newzi,gi).std()),.1);pi=postprocess(center(newzi,gi)/scale,gi,'conditional',.6);pv=postprocess(center(newzv,gv)/scale,gv,'conditional',.6);di=center(logit(np.clip(pi,1e-6,1-1e-6))-logit(np.clip(ib[83],1e-6,1-1e-6)),gi);dv=center(logit(np.clip(pv,1e-6,1-1e-6))-logit(np.clip(ob[83],1e-6,1-1e-6)),gv);ds=max(float(di.std()),.1);di/=ds;dv/=ds
  X=np.column_stack([di,.5*(2*e[:,6]-1)*di])*(4*qi*(1-qi))[:,None];V=np.column_stack([dv,.5*(2*ev[:,6]-1)*dv])*(4*qv*(1-qv))[:,None];offset=logit(np.clip(qi,1e-6,1-1e-6));offsetv=logit(np.clip(qv,1e-6,1-1e-6));y=d['label'][tr];pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0);reg=1/(.01*len(tr))
  def obj(beta):
   z=offset+X@beta;prob=expit(z);margin=z[pos,None]-z[neg][None];dm=-expit(-margin)/margin.size;gr=np.zeros(len(y));gr[pos]=dm.sum(1);gr[neg]=-dm.sum(0);grad=.5*(prob-y)/len(y)+.5*gr;loss=.5*np.mean(np.logaddexp(0,z)-y*z)+.5*np.logaddexp(0,-margin).mean()+.5*reg*np.dot(beta,beta)
   return float(loss),X.T@grad+reg*beta
  opt=minimize(obj,np.zeros(2),jac=True,method='L-BFGS-B',options=dict(maxiter=500,ftol=1e-12,gtol=1e-8));assert opt.success;artifact=dict(beta=opt.x,scale=scale,delta_scale=ds,frozen_teacher_fold=k,protocol=spec);path=OUT/f'fold{k}.joblib';joblib.dump(artifact,path);saved=joblib.load(path);q=expit(offsetv+V@opt.x);replay=expit(offsetv+V@saved['beta']);err=float(abs(q-replay).max());assert err<1e-12;result[full['model_row_to_full_row'][va]]=q;records.append(dict(fold=k,beta=opt.x.tolist(),reload_error=err));print('FROZEN RESIDUAL',k,'/5 COMPLETE',flush=True)
 y=full['label'];subjects=full['subject'];people=np.unique(subjects);delta=pair_matrix(y,result,subjects,people)-pair_matrix(y,anchor,subjects,people);rng=np.random.default_rng(1031);counts=np.zeros((2000,len(people)))
 for j in range(2000):
  for flag in ['HC','DEP']:
   ix=np.flatnonzero(np.char.startswith(people,flag));counts[j]+=np.bincount(rng.choice(ix,len(ix),replace=True),minlength=len(people))
 boot=np.einsum('bi,ij,bj->b',counts,delta,counts,optimize=True)/counts.sum(1)**2
 metrics=dict(primary_pooled_auc=float(roc_auc_score(y,result)),baseline_pooled_auc=float(roc_auc_score(y,anchor)),DEP_auc=float(roc_auc_score(y[full['diagnosis']==1],result[full['diagnosis']==1])),paired_person_delta_95CI=np.quantile(boot,[.025,.975]).tolist(),folds=records,cohort_reused=True,independent_confirmation=False,baseline_preserved=True,official_submission=False)
 np.savez_compressed(OUT/'oof.npz',probability=result,subjects=subjects);driver.save(OUT/'summary.json',metrics);print(json.dumps(metrics),flush=True)
if __name__=='__main__':main()
