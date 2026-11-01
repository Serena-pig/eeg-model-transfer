"""Source-inner feature-view selection for within-person pair comparisons.
Compare to the frozen selected comparator, not to a weaker graph control.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import json,time,shutil
import numpy as np,joblib
from scipy.stats import rankdata,norm
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
import driver
from common import center
import pair_linear_range_20261101 as base
OUT=driver.JOB/'pair_views_20261101';OUT.mkdir(exist_ok=True)
VIEWS=['center','within8_z','within8_rank','relative','regional','spatial']
CONFIGS=[dict(kind='linear',view=v,C=c) for v in VIEWS for c in [.03,.3,3.]]
REGIONS=[[0,1,2,3,4,5,6],[7,8,9,10,11],[12,16,17,21,22,26],[13,14,15,18,19,20],[23,24,25],[27,28,29]]
LEFT=[0,2,3,7,8,12,13,17,18,22,23,27];RIGHT=[1,6,5,11,10,16,15,21,20,26,25,29]

class Data(base.Data):
 def __init__(self):
  super().__init__();original=self.x.copy();self.views={'center':original,'within8_z':center(original,self.groups,zscore=True)};rank=np.empty_like(original)
  for g in np.unique(self.groups):
   ix=np.flatnonzero(self.groups==g);rank[ix]=norm.ppf((rankdata(original[ix],axis=0,method='average')-.5)/len(ix))
  self.views['within8_rank']=rank
  with np.load(driver.ART/'features.npz') as a:relative=a['relative'].reshape(len(original),-1)
  self.views['relative']=center(np.column_stack([original,relative]),self.groups);blocks=original.reshape(len(original),3,6,30);assert sorted(sum(REGIONS,[]))==list(range(30));pooled=np.stack([blocks[:,:,:,ix].mean(-1) for ix in REGIONS],-1).reshape(len(original),-1);asym=(blocks[:,:,:,LEFT]-blocks[:,:,:,RIGHT]).reshape(len(original),-1);self.views['regional']=center(np.column_stack([pooled,asym]),self.groups);self.views['spatial']=center((blocks-blocks.mean(-1,keepdims=True)).reshape(len(original),-1),self.groups)
  for x in self.views.values():assert np.isfinite(x).all()
 def select(self,view):self.x=self.views[view]

def train(data,tr,config):
 data.select(config['view']);a=base.train(data,tr,config);a['config']=dict(config);return a

def predict(data,va,artifact):
 data.select(artifact['config']['view']);return base.predict(data,va,artifact)

def main():
 threadpool_limits(1);spec=dict(configs=CONFIGS,region_index_groups=REGIONS,left_right_pairs=list(zip(LEFT,RIGHT)),region_order_note='Project existing CHANNELS ordering assumption; pooled features, not physiological causal claims',selection='max pooled source3inner AUC; require .001 over existing pair comparator else old_pair',primary_metric='unweighted_pooled_auc',source_scaler_only=True,query_group_transform='own8records; average tied ranks; no labels',public_training=False,holdout_used=False,script_sha256=driver.sha(__file__))
 driver.save(OUT/'protocol.json',spec);data=Data();d=data.d;full=data.full;selections=[]
 for f in driver.protocol()['folds']:
  k=f['fold'];predictions=[[] for c in CONFIGS];labels=[];anchor=[]
  for part in range(3):
   split=f['inner'][part];tr=np.flatnonzero(np.isin(d['subject'],split['training_subjects']));va=np.flatnonzero(np.isin(full['subject'],split['validation_subjects']));assert set(d['subject'][tr]).isdisjoint(full['subject'][va]);labels.append(full['label'][va])
   with np.load(driver.JOB/f'pair_linear_range_20261101/selected_fold{k}_inner{part}.npz') as a:assert np.array_equal(a['validation'],va);anchor.append(a['probability'])
   for j,config in enumerate(CONFIGS):artifact=train(data,tr,config);predictions[j].append(predict(data,va,artifact))
   np.savez_compressed(OUT/f'fold{k}_inner{part}_all.npz',validation=va,probability=np.stack([p[-1] for p in predictions]));print('PAIR VIEWS INNER',k,part,'/3 COMPLETE',flush=True)
  yy=np.concatenate(labels);before=float(roc_auc_score(yy,np.concatenate(anchor)));scores=[float(roc_auc_score(yy,np.concatenate(q))) for q in predictions];j=int(np.argmax(scores));chosen=CONFIGS[j] if scores[j]>=before+.001 else 'old_pair';record=dict(fold=k,chosen=chosen,existing_pair_source_auc=before,source_candidate_auc=scores,configs=CONFIGS);selections.append(record);driver.save(OUT/f'fold{k}_selection.json',record)
  for part in range(3):
   if chosen=='old_pair':shutil.copy2(driver.JOB/f'pair_linear_range_20261101/selected_fold{k}_inner{part}.npz',OUT/f'selected_fold{k}_inner{part}.npz')
   else:
    with np.load(OUT/f'fold{k}_inner{part}_all.npz') as a:va=a['validation'];q=a['probability'][j]
    np.savez_compressed(OUT/f'selected_fold{k}_inner{part}.npz',validation=va,probability=q)
  va=np.flatnonzero(np.isin(full['subject'],f['validation_subjects']))
  if chosen=='old_pair':shutil.copy2(driver.JOB/f'pair_linear_range_20261101/selected_fold{k}_outer.npz',OUT/f'selected_fold{k}_outer.npz')
  else:
   tr=np.flatnonzero(np.isin(d['subject'],f['training_subjects']));artifact=train(data,tr,chosen);path=OUT/f'fold{k}.joblib';joblib.dump(artifact,path,compress=3);q=predict(data,va,artifact);saved=joblib.load(path);per=np.random.default_rng(k).permutation(len(va));error=float(abs(predict(data,va[per],saved)-q[per]).max());assert error<1e-12;driver.save(OUT/f'fold{k}_native_check.json',dict(reload_permutation_error=error,source_only=True));np.savez_compressed(OUT/f'selected_fold{k}_outer.npz',validation=va,probability=q)
  print('PAIR VIEWS OUTER',k,'/5 COMPLETE',chosen,flush=True)
 driver.save(OUT/'completion.json',dict(selections=selections,all5outer3inner_complete=True,public_training=False,holdout_used=False,official_submissions=0));print('PAIR VIEWS COMPLETE',flush=True)
if __name__=='__main__':main()
