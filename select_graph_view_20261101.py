"""Pick raw vs aligned graph from source3inner predictions, never outer AUC.
Assemble matching saved outer fusion heads; no new/full-meta-CV claim.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import time,json
import numpy as np
from sklearn.metrics import roc_auc_score
import driver
from postflight import pair_matrix
OUT=driver.JOB/'graph_view_selection_20261101';OUT.mkdir(exist_ok=True)
VIEWS={'raw':'connectivity_graph_20261101','aligned':'aligned_graph_20261101'}
def main():
 driver.save(OUT/'protocol.json',dict(selection='max unweighted pooled AUC of source3inner forecasts; tie prefers raw',views=VIEWS,source_only=True,outer_not_used_for_choice=True,primary_metric='unweighted_pooled_auc',cohort_reused=True,script_sha256=driver.sha(__file__)))
 with np.load(driver.CONTROL/'full_evaluation_rows.npz') as a:full={k:a[k] for k in a.files}
 choices=[]
 for f in driver.protocol()['folds']:
  scores={}
  for view,directory in VIEWS.items():
   yy=[];qq=[]
   for part in range(3):
    path=driver.JOB/directory/f'selected_fold{f["fold"]}_inner{part}.npz'
    while True:
     try:
      with np.load(path) as a:indices=a['validation'];q=a['probability'];assert np.isfinite(q).all()
      break
     except (OSError,ValueError,EOFError):time.sleep(3)
    yy.append(full['label'][indices]);qq.append(q)
   scores[view]=float(roc_auc_score(np.concatenate(yy),np.concatenate(qq)))
  chosen=max(VIEWS,key=lambda k:scores[k]);record=dict(fold=f['fold'],chosen=chosen,source_scores=scores);choices.append(record);driver.save(OUT/f'fold{f["fold"]}_selection.json',record);print('SOURCE GRAPH CHOICE',record,flush=True)
 for folder in ['connectivity_graph_fusion_20261101','aligned_graph_fusion_20261101']:
  while not (driver.JOB/folder/'summary.json').exists():time.sleep(3)
 with np.load(driver.JOB/'pooled_rank_bag_20261030/subbag16_rawmean_calibrated_oof.npz') as a:anchor=a['probability']
 q=anchor.copy()
 for choice,f in zip(choices,driver.protocol()['folds']):
  directory='connectivity_graph_fusion_20261101' if choice['chosen']=='raw' else 'aligned_graph_fusion_20261101'
  with np.load(driver.JOB/directory/'subbag16_rawmean_calibrated_oof.npz') as a:probability=a['probability']
  mask=np.isin(full['subject'],f['validation_subjects']);q[mask]=probability[mask]
 y=full['label'];people=np.unique(full['subject']);delta=pair_matrix(y,q,full['subject'],people)-pair_matrix(y,anchor,full['subject'],people);rng=np.random.default_rng(1101);counts=np.zeros((5000,len(people)))
 for j in range(len(counts)):
  for name in ['HC','DEP']:
   ix=np.flatnonzero(np.char.startswith(people,name));counts[j]+=np.bincount(rng.choice(ix,len(ix),replace=True),minlength=len(people))
 boot=np.einsum('bi,ij,bj->b',counts,delta,counts,optimize=True)/counts.sum(1)**2
 result=dict(primary_pooled_auc=float(roc_auc_score(y,q)),baseline_pooled_auc=float(roc_auc_score(y,anchor)),DEP_auc=float(roc_auc_score(y[full['diagnosis']==1],q[full['diagnosis']==1])),source_choices=choices,paired_person_delta_95CI=np.quantile(boot,[.025,.975]).tolist(),cohort_reused=True,independent_confirmation=False,public_signal_training=False,holdout_used=False,official_submission=False)
 np.savez_compressed(OUT/'oof.npz',probability=q);driver.save(OUT/'summary.json',result);print('SOURCE VIEW POLICY COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main()
