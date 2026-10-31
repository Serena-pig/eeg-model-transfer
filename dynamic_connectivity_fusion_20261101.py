"""Add source-inner-selected cross-crop consistency CNN, preserving all87 original members.
Refit original16 subbag ranking fusion; no public signals or outer selection.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import json,traceback,time
from concurrent.futures import ThreadPoolExecutor
import numpy as np,joblib
from scipy.special import expit,softmax,logsumexp
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.special import logit
from threadpoolctl import threadpool_limits
import driver
from common import batches,center
from cached_fixed_fusion_20261030 import features
from context_rank_v34 import fit_context,context_features
from select_ensemble import ASSIGNMENTS,postprocess
from inner_calibration_round import calibrate
from postflight import pair_matrix
OUT=driver.JOB/'dynamic_connectivity_fusion_20261101';OUT.mkdir(exist_ok=True)
BAGS=16
NAMES=['subbag16_rawmean','subbag16_groupmean']
def main():
 threadpool_limits(limits=1);p=driver.protocol();spec=dict(variants=NAMES,bags_fixed=BAGS,primary_metric='unweighted_pooled_auc',primary_variant='subbag16_rawmean_calibrated',new_expert='static graph retained; source-inner-selected local connectivity expert',additional_neural_columns=[87,88],old_CNN_retained=True,real_public_data_used=False,subbag_fraction_fixed=.8,resampling='Subject clusters with all40windows; diagnosis stratum masses preserve actual proportions',seed_fixed=1030,C_inherited=.01,conditional_mix_fixed=.5,ranking_loss='Exact weighted all-pairs logistic loss on conditional-posterior logits',total_weight_per_bag='Original source N',warm_start='Full source fit, no out-of-bag validation claimed',CPU_workers=1,no_outer_parameter_search=True,base_models_retrained=0,calibration='Previous source-inner policy inherited',script_sha256=driver.sha(__file__))
 if (OUT/'protocol.json').exists():assert json.loads((OUT/'protocol.json').read_text())==spec
 else:driver.save(OUT/'protocol.json',spec)
 with np.load(driver.ART/'features.npz') as a:d={k:a[k] for k in ['subject','trial','crop','label','diagnosis','train','powers','quality']}
 with np.load(driver.CONTROL/'full_evaluation_rows.npz') as a:full={k:a[k] for k in a.files}
 with np.load(driver.JOB/'pooled_rank_bag_20261030/subbag16_rawmean_calibrated_oof.npz') as a:anchor=a['probability']
 out={n:anchor.copy() for n in NAMES+[n+'_calibrated' for n in NAMES]};audits=[];baseix=np.r_[np.arange(83),np.arange(104,107)]
 for f in p['folds']:
  decision_path=driver.JOB/f'dynamic_connectivity_20261101/fold{f["fold"]}_selection.json'
  while not decision_path.exists():time.sleep(3)
  decision=json.loads(decision_path.read_text())
  if decision['chosen']=='old_static':
   import shutil
   src=driver.JOB/f'connectivity_graph_fusion_20261101/fold{f["fold"]}.joblib';shutil.copy2(src,OUT/f'fold{f["fold"]}.joblib');mask=np.isin(full['subject'],f['validation_subjects'])
   for name in out:
    with np.load(driver.JOB/f'connectivity_graph_fusion_20261101/{name}_oof.npz') as data:out[name][mask]=data['probability'][mask]
   driver.save(OUT/f'fold{f["fold"]}_reused.json',dict(reason='No source-inner local graph gain, keep frozen88member candidate',source_sha256=driver.sha(src)));audits.append(dict(fold=f['fold'],reused_static_fusion=True));print('LOCAL CONNECTIVITY FUSION',f['fold'],'REUSED STATIC',flush=True);continue
  # Await complete upstream arrays; fit ready folds while next CNN trains.
  paths=[driver.JOB/f'dynamic_connectivity_20261101/selected_fold{f["fold"]}_{part}.npz' for part in ['inner0','inner1','inner2','outer']]
  while True:
   try:
    for path in paths:
     with np.load(path) as payload:assert np.isfinite(payload['probability']).all()
    break
   except (OSError,ValueError,EOFError):time.sleep(3)
  with np.load(driver.JOB/f'cached_group_diagnostics_20261030/fold{f["fold"]}_aligned_banks.npz') as a:tr=a['inner_indices'];va=a['outer_indices'];ib=a['inner_probability'][baseix];ob=a['outer_probability'][baseix]
  lookup={v:j for j,v in enumerate(tr)};zin=np.zeros(len(tr))
  for part in range(3):
   with np.load(driver.JOB/f'aligned_covariance_svm_20261030/fold{f["fold"]}_inner{part}.npz') as a:zin[[lookup[v] for v in a['validation']]]=a['logits']
  with np.load(driver.JOB/f'aligned_covariance_svm_20261030/fold{f["fold"]}_outer.npz') as a:assert np.array_equal(a['validation'],va);zout=a['logits']
  # Replace only the3neural source-OOF columns, using their own clean-inner scales.
  reverse=np.full(len(full['label']),-1,dtype=int);reverse[full['model_row_to_full_row']]=np.arange(len(d['label']));tta_scales=[]
  for column,mode in [(83,'de_cnn')]:
   zi=np.zeros(len(tr));root=driver.JOB/'dynamic_connectivity_20261101'
   for part in range(3):
    with np.load(root/f'selected_fold{f["fold"]}_inner{part}.npz') as a:
     good=reverse[a['validation']]>=0;rrr=reverse[a['validation'][good]];zi[[lookup[v] for v in rrr]]=logit(np.clip(a['probability'][good],1e-6,1-1e-6))
   with np.load(root/f'selected_fold{f["fold"]}_outer.npz') as a:
    good=reverse[a['validation']]>=0;assert np.array_equal(reverse[a['validation'][good]],va);zv=logit(np.clip(a['probability'][good],1e-6,1-1e-6))
   gi=batches(d)[tr];go=batches(d)[va];scale=max(float(center(zi,gi).std()),.1);tta_scales.append(scale);newib=postprocess(center(zi,gi)/scale,gi,'conditional',.6);newob=postprocess(center(zv,go)/scale,go,'conditional',.6)
  # Keep static and temporal predictions as distinct sources.
  rawzi=np.zeros(len(tr))
  for part in range(3):
   with np.load(driver.JOB/f'connectivity_graph_20261101/selected_fold{f["fold"]}_inner{part}.npz') as a:
    good=reverse[a['validation']]>=0;rows=reverse[a['validation'][good]];rawzi[[lookup[v] for v in rows]]=logit(np.clip(a['probability'][good],1e-6,1-1e-6))
  with np.load(driver.JOB/f'connectivity_graph_20261101/selected_fold{f["fold"]}_outer.npz') as a:
   good=reverse[a['validation']]>=0;assert np.array_equal(reverse[a['validation'][good]],va);rawzv=logit(np.clip(a['probability'][good],1e-6,1-1e-6))
  raw_scale=max(float(center(rawzi,gi).std()),.1);rawin=postprocess(center(rawzi,gi)/raw_scale,gi,'conditional',.6);rawout=postprocess(center(rawzv,go)/raw_scale,go,'conditional',.6)
  base=joblib.load(driver.JOB/f'pooled_weight_fusion_20261030/fold{f["fold"]}.joblib');sd=base['SPD_source_scale'];gin=batches(d)[tr];gout=batches(d)[va];ib=np.vstack([ib,postprocess(center(zin,gin)/sd,gin,'conditional',.6)]);ob=np.vstack([ob,postprocess(center(zout,gout)/sd,gout,'conditional',.6)]);ib=np.vstack([ib,rawin,newib]);ob=np.vstack([ob,rawout,newob]);source=dict(d);source['label']=np.full_like(d['label'],-1);source['diagnosis']=np.full_like(d['diagnosis'],-1);source['diagnosis'][tr]=d['diagnosis'][tr];context,e=fit_context(source,tr);blind=dict(d);blind['label']=np.full_like(d['label'],-1);blind['diagnosis']=np.full_like(d['diagnosis'],-1);ev=context_features(blind,va,context);base['scaler']=StandardScaler().fit(np.column_stack([features(ib,gin),e]));base['context']=context;base['consistency_neural_source_scales']=tta_scales;base['static_graph_source_scale']=raw_scale;base['local_connectivity_strategy']=json.loads((driver.JOB/f'dynamic_connectivity_20261101/fold{f["fold"]}_selection.json').read_text())['chosen'];base['protocol']=spec;base.pop('models',None);xx=base['scaler'].transform(np.column_stack([features(ib,gin),e]));vv=base['scaler'].transform(np.column_stack([features(ob,gout),ev]));gx=np.column_stack([xx,.5*(2*e[:,6]-1)[:,None]*xx[:,[83,84,85,86,87,88]]]);gv=np.column_stack([vv,.5*(2*ev[:,6]-1)[:,None]*vv[:,[83,84,85,86,87,88]]]);initial=joblib.load(driver.JOB/f'pooled_posterior_rank_20261030/fold{f["fold"]}.joblib')['models']['pooled_posterior_pair'];old_initial=initial.copy();initial=np.zeros(195);initial[:87]=old_initial[:87];initial[89:176]=old_initial[87:174];initial[178:188]=old_initial[174:184];initial[188:192]=old_initial[184:188];initial[-1]=old_initial[-1];y=d['label'][tr];rr=np.stack([np.flatnonzero(gin==g) for g in np.unique(gin)]);reg=1/(.01*len(y));people=np.unique(d['subject'][tr]);indices=[np.flatnonzero(d['subject'][tr]==s) for s in people];diag=np.array([d['diagnosis'][tr[ix[0]]] for ix in indices]);A=ASSIGNMENTS;mode=json.loads((driver.JOB/f'inner_calibration_20261028/fold{f["fold"]}_selection.json').read_text())['chosen'];models={};weight_records={}
  def fit_head(roww):
   wn=roww/roww.sum();gw=roww[rr].mean(1);gw/=gw.sum();positive=np.flatnonzero(y==1);negative=np.flatnonzero(y==0);pairw=wn[positive,None]*wn[negative][None];pairw/=pairw.sum()
   def obj(theta):
    beta=theta[:-1];z=gx@beta+theta[-1];ss=z[rr]@A.T;post=softmax(ss,axis=1);prob=post@A;safe=np.clip(prob,1e-8,1-1e-8);lp=np.log(safe)-np.log1p(-safe);flat=np.zeros(len(y));flat[rr.ravel()]=lp.ravel();margin=flat[positive,None]-flat[negative][None];risk=np.sum(pairw*np.logaddexp(0,-margin));dm=-pairw*expit(-margin);gl=np.zeros(len(y));gl[positive]=dm.sum(1);gl[negative]=-dm.sum(0);cov=np.einsum('ga,ai,aj->gij',post,A,A,optimize=True)-prob[:,:,None]*prob[:,None,:];v=gl[rr]/(safe*(1-safe));v[(prob<=1e-8)|(prob>=1-1e-8)]=0.;pg=np.zeros(len(y));pg[rr.ravel()]=np.einsum('gij,gi->gj',cov,v,optimize=True).ravel();closs=np.dot(gw,(logsumexp(ss,axis=1)-(z[rr]*y[rr]).sum(1))/8);ce=np.zeros(len(y));ce[rr.ravel()]=((prob-y[rr])*gw[:,None]/8).ravel();gradient=.5*ce+.5*pg
    return float(.5*closs+.5*risk+.5*reg*np.dot(beta,beta)),np.r_[gx.T@gradient+reg*beta,gradient.sum()]
   if np.array_equal(roww,weights[0]):
    grad=obj(initial)[1]
    for j in [0,len(initial)//2,len(initial)-1]:
     aa=initial.copy();bb=initial.copy();aa[j]+=1e-6;bb[j]-=1e-6;assert abs((obj(aa)[0]-obj(bb)[0])/2e-6-grad[j])<1e-5
   fit=minimize(obj,initial.copy(),jac=True,method='L-BFGS-B',options=dict(maxiter=1000,ftol=1e-12,gtol=1e-7));assert fit.success,fit.message;return fit.x
  for method in ['subbag16']:
   weights=[];rng=np.random.default_rng(1030+f['fold']*100+(0 if method=='bootstrap16' else 50))
   for j in range(BAGS):
    w=np.zeros(len(y))
    for v in (0,1):
     pool=np.flatnonzero(diag==v)
     if method=='bootstrap16':chosen=rng.choice(pool,len(pool),replace=True);counts=np.bincount(chosen,minlength=len(people)).astype('float64')
     else:chosen=rng.choice(pool,max(2,int(round(.8*len(pool)))),replace=False);counts=np.zeros(len(people));counts[chosen]=len(pool)/len(chosen)
     for k in pool:w[indices[k]]=counts[k]
    assert np.isclose(w.sum(),len(y));weights.append(w)
   with ThreadPoolExecutor(max_workers=1) as executor:thetas=np.array(list(executor.map(fit_head,weights)))
   models[method]=thetas;weight_records[method]=np.array(weights);raw=np.mean([expit(gv@t[:-1]+t[-1]) for t in thetas],axis=0);group=np.mean([postprocess(gv@t[:-1]+t[-1],gout,'conditional',1.) for t in thetas],axis=0)
   for agg,q in [('rawmean',raw),('groupmean',group)]:
    name=method+'_'+agg;out[name][full['model_row_to_full_row'][va]]=np.round(q,12);out[name+'_calibrated'][full['model_row_to_full_row'][va]]=np.round(q if agg=='groupmean' else calibrate(q,gout,mode),12)
   print('LOCAL CONNECTIVITY FUSION',f['fold'],method,'16complete',flush=True)
  path=OUT/f'fold{f["fold"]}.joblib';joblib.dump(dict(models=models,weights=weight_records,base_artifact=base,context=context,source_subjects=f['training_subjects'],protocol=spec),path,compress=3);saved=joblib.load(path);per=np.random.default_rng(f['fold']).permutation(len(va))
  for method,thetas in models.items():
   assert np.array_equal(saved['models'][method],thetas);a=np.mean([postprocess(gv@t[:-1]+t[-1],gout,'conditional',1.) for t in thetas],axis=0);r=np.mean([postprocess(gv[per]@t[:-1]+t[-1],np.char.add('opaque_',gout[per]),'conditional',1.) for t in saved['models'][method]],axis=0);error=float(abs(r-a[per]).max());assert error<1e-9;audits.append(dict(fold=f['fold'],method=method,native_opaque_order_delta=error))
  print('LOCAL CONNECTIVITY FUSION OUTER',f['fold'],'/5',flush=True)
 y=full['label'];g=full['subject'];dep=full['diagnosis']==1;ids=np.unique(g);ref=pair_matrix(y,anchor,g,ids);rng=np.random.default_rng(1030);counts=np.zeros((2000,len(ids)))
 for j in range(len(counts)):
  for mask in [np.char.startswith(ids,'DEP'),np.char.startswith(ids,'HC')]:
   ix=np.flatnonzero(mask);counts[j]+=np.bincount(rng.choice(ix,len(ix),replace=True),minlength=len(ids))
 # Preserve original unweighted pooled AUC bootstrap.
 results={}
 for name,q in out.items():
  delta=pair_matrix(y,q,g,ids)-ref;boot=np.einsum('bi,ij,bj->b',counts,delta,counts,optimize=True)/counts.sum(1)**2;results[name]=dict(pooled_auc=float(roc_auc_score(y,q)),equal_clinical_mass_pooled_auc=float(roc_auc_score(y,q,sample_weight=np.where(dep,2.,1.))),DEP_auc=float(roc_auc_score(y[dep],q[dep])),paired_person_delta_95CI=np.quantile(boot,[.025,.975]).tolist());np.savez_compressed(OUT/(name+'_oof.npz'),labels=y,subjects=g,probability=q)
 driver.save(OUT/'summary.json',dict(results=results,anchor_auc=float(roc_auc_score(y,anchor)),anchor_equal_clinical_mass_pooled_auc=float(roc_auc_score(y,anchor,sample_weight=np.where(dep,2.,1.))),paired_person_interval_metric='unweighted_pooled_auc',baseline_preserved=True,cohort_reused=True,multiple_search_uncorrected=True,official_score_unknown=True));driver.save(OUT/'verification.json',dict(checks_passed=True,all5outer_complete=True,source_only_patient_resampling=True,no_out_of_bag_validation_claimed=True,calibration_policy_inherited=True,audits=audits,holdout_and_official_not_read=True,submissions_performed=False));print('LOCAL CONNECTIVITY FUSION COMPLETE',json.dumps(results),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:driver.save(OUT/'failure.json',dict(traceback=traceback.format_exc()));raise
