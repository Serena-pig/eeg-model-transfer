"""Within-person physiological pair comparator; odd Fourier features + Borda.
Raw label-free features, source3inner selection, no graph embedding dependence.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import json,time
import numpy as np,joblib
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
import driver
from common import batches,center
OUT=driver.JOB/'pair_difference_20261101';OUT.mkdir(exist_ok=True)
CONFIGS=[dict(kind='linear',C=c) for c in [.03,.3]]+[dict(kind='odd_rff',gamma=g,C=c) for g in [.25,1.] for c in [.03,.3]]

class Data:
 def __init__(self):
  with np.load(driver.ART/'features.npz') as a:self.d={k:a[k] for k in ['subject','trial','crop','label','diagnosis']};power=a['powers'][:,:6].reshape(len(a['label']),-1);variation=a['local'].std(1).reshape(len(a['label']),-1)
  with np.load(driver.JOB/'aligned_covariance_20261030/features.npz') as a:assert np.array_equal(a['subject'],self.d['subject']);assert np.array_equal(a['trial'],self.d['trial']);assert np.array_equal(a['crop'],self.d['crop']);diagonal=a['diagonal']
  with np.load(driver.CONTROL/'full_evaluation_rows.npz') as a:self.full={k:a[k] for k in a.files}
  self.reverse=np.full(len(self.full['label']),-1,dtype=int);self.reverse[self.full['model_row_to_full_row']]=np.arange(len(self.d['label']));self.groups=batches(self.d);self.x=center(np.column_stack([power,variation,diagonal]).astype(float),self.groups)

def phi(difference,projection):return difference/np.sqrt(difference.shape[-1]) if projection is None else np.sqrt(2/projection.shape[1])*np.sin(difference@projection)

def train(data,tr,config):
 scaler=StandardScaler().fit(data.x[tr]);x=scaler.transform(data.x[tr]);y=data.d['label'][tr];group=data.groups[tr];differences=[]
 for key in np.unique(group):
  indices=np.flatnonzero(group==key);assert len(indices)==8;pos=indices[y[indices]==1];neg=indices[y[indices]==0];assert len(pos)==len(neg)==4;differences.append((x[pos,None]-x[neg][None]).reshape(-1,x.shape[1]))
 differences=np.concatenate(differences);projection=None
 if config['kind']=='odd_rff':projection=np.random.default_rng(110181).normal(size=(x.shape[1],256))*np.sqrt(2*config['gamma']/x.shape[1])
 features=phi(differences,projection);model=LogisticRegression(C=config['C'],fit_intercept=False,solver='lbfgs',max_iter=1000,tol=1e-8).fit(np.vstack([features,-features]),np.r_[np.ones(len(features)),np.zeros(len(features))]);assert model.n_iter_[0]<1000
 return dict(scaler=scaler,projection=projection,model=model,config=config,source_subjects=np.unique(data.d['subject'][tr]).tolist())

def predict(data,va,artifact):
 valid=data.reverse[va]>=0;indices=data.reverse[va[valid]];x=artifact['scaler'].transform(data.x[indices]);group=data.groups[indices];q=np.full(len(va),.5);values=np.zeros(len(indices))
 for key in np.unique(group):
  rr=np.flatnonzero(group==key);assert len(rr)==8;difference=(x[rr,None]-x[rr][None]).reshape(64,-1);features=phi(difference,artifact['projection']);wins=expit(artifact['model'].decision_function(features)).reshape(8,8);assert np.max(abs(wins+wins.T-1))<1e-12;values[rr]=(wins.sum(1)-.5)/7
 q[valid]=values;return q

def main():
 threadpool_limits(1);spec=dict(configs=CONFIGS,features='centered6band powers+local2s variability+aligned covariance diagonal',pair_labels='Within source patient/crop positive-minus-neutral, both orientations',antisymmetry='odd sine features and zero intercept; p(i>j)+p(j>i)=1',inference='mean predicted wins against other7trials; no query labels',primary_metric='unweighted_pooled_auc',selection='max pooled3sourceinner AUC, require .001 over old graph else fallback',seed=110181,public_data_used=False,holdout_used=False,script_sha256=driver.sha(__file__));driver.save(OUT/'protocol.json',spec);data=Data();d=data.d;full=data.full;selections=[]
 for f in driver.protocol()['folds']:
  k=f['fold'];predictions=[[] for _ in CONFIGS];anchor=[];labels=[]
  for part in range(3):
   split=f['inner'][part];tr=np.flatnonzero(np.isin(d['subject'],split['training_subjects']));va=np.flatnonzero(np.isin(full['subject'],split['validation_subjects']));assert set(d['subject'][tr]).isdisjoint(full['subject'][va]);labels.append(full['label'][va])
   with np.load(driver.JOB/f'connectivity_graph_20261101/selected_fold{k}_inner{part}.npz') as a:assert np.array_equal(a['validation'],va);anchor.append(a['probability'])
   for j,c in enumerate(CONFIGS):artifact=train(data,tr,c);predictions[j].append(predict(data,va,artifact))
   np.savez_compressed(OUT/f'fold{k}_inner{part}_all.npz',validation=va,probability=np.stack([v[-1] for v in predictions]));print('PAIR COMPARATOR INNER',k,part,'/3 COMPLETE',flush=True)
  yy=np.concatenate(labels);baseline=float(roc_auc_score(yy,np.concatenate(anchor)));scores=[float(roc_auc_score(yy,np.concatenate(v))) for v in predictions];j=int(np.argmax(scores));chosen=CONFIGS[j] if scores[j]>=baseline+.001 else 'old_static';record=dict(fold=k,chosen=chosen,old_graph_source_auc=baseline,source_candidate_auc=scores,configs=CONFIGS);selections.append(record);driver.save(OUT/f'fold{k}_selection.json',record)
  for part in range(3):
   if chosen=='old_static':
    with np.load(driver.JOB/f'connectivity_graph_20261101/selected_fold{k}_inner{part}.npz') as a:va=a['validation'];q=a['probability']
   else:
    with np.load(OUT/f'fold{k}_inner{part}_all.npz') as a:va=a['validation'];q=a['probability'][j]
   np.savez_compressed(OUT/f'selected_fold{k}_inner{part}.npz',validation=va,probability=q)
  va=np.flatnonzero(np.isin(full['subject'],f['validation_subjects']))
  if chosen=='old_static':
   with np.load(driver.JOB/f'connectivity_graph_20261101/selected_fold{k}_outer.npz') as a:assert np.array_equal(a['validation'],va);q=a['probability']
  else:
   tr=np.flatnonzero(np.isin(d['subject'],f['training_subjects']));artifact=train(data,tr,chosen);path=OUT/f'fold{k}.joblib';joblib.dump(artifact,path,compress=3);q=predict(data,va,artifact);saved=joblib.load(path);per=np.random.default_rng(k).permutation(len(va));error=float(abs(predict(data,va[per],saved)-q[per]).max());assert error<1e-12;driver.save(OUT/f'fold{k}_native_check.json',dict(reload_permutation_error=error,query_labels_used=False,source_subjects=artifact['source_subjects']))
  np.savez_compressed(OUT/f'selected_fold{k}_outer.npz',validation=va,probability=q);print('PAIR COMPARATOR OUTER',k,'/5 COMPLETE',chosen,flush=True)
 driver.save(OUT/'completion.json',dict(selections=selections,all5outer3inner_complete=True,public_data_used=False,holdout_used=False,official_submissions=0));print('PAIR COMPARATOR COMPLETE',flush=True)
if __name__=='__main__':main()
