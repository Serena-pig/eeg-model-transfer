"""Descriptive error audit; no selection, public fitting or deployment changes.
Within-person probe has a different label budget and cannot be submitted.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import json,csv,warnings
import numpy as np
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
import driver
from common import batches,center
OUT=driver.JOB/'error_cause_audit_20261031';OUT.mkdir(exist_ok=True)

def rho_ci(x,y):
 x=np.asarray(x);y=np.asarray(y);rng=np.random.default_rng(1031)
 def corr(a,b):
  if np.std(a)==0 or np.std(b)==0:return None
  return float(spearmanr(a,b).statistic)
 r=corr(x,y);boot=[]
 for _ in range(2000):
  ix=rng.integers(len(x),size=len(x));z=corr(x[ix],y[ix])
  if z is not None and np.isfinite(z):boot.append(z)
 return dict(rho=r,person_bootstrap_95CI=np.quantile(boot,[.025,.975]).tolist() if boot else None,n_people=len(x),causal=False)

def main():
 threadpool_limits(1)
 with np.load(driver.ART/'features.npz') as a:d={k:a[k] for k in ['subject','trial','crop','label','diagnosis','powers','relative','sequence']}
 with np.load(driver.CONTROL/'full_evaluation_rows.npz') as a:full={k:a[k] for k in a.files}
 with np.load(driver.JOB/'pooled_rank_bag_20261030/subbag16_rawmean_calibrated_oof.npz') as a:q=a['probability'].copy()
 g=full['subject'];y=full['label'];people=np.unique(g);mapping=full['model_row_to_full_row'];quality=full['quality'];de=d['powers'][:,:6].reshape(len(mapping),-1)
 # Label-free context distance to the corresponding outer training people.
 means={s:de[d['subject']==s].mean(0) for s in np.unique(d['subject'])};dist={}
 for f in driver.protocol()['folds']:
  tr=[s for s in f['training_subjects'] if s in means];xx=np.stack([means[s] for s in tr]);scaler=StandardScaler().fit(xx);xx=scaler.transform(xx)
  for s in f['validation_subjects']:
   if s in means:dist[s]=float(np.sqrt(np.min(np.mean((xx-scaler.transform(means[s][None]))**2,axis=1))))
 records=[];probe=q.copy();group=batches(d)
 raw=np.column_stack([de,d['relative'].reshape(len(mapping),-1),d['sequence'].std(1).reshape(len(mapping),-1)])
 xx=center(raw,group)
 for s in people:
  mask=g==s;ix=np.flatnonzero(d['subject']==s);auc=float(roc_auc_score(y[mask],q[mask]));cropauc=[float(roc_auc_score(y[mask&(full['crop']==c)],q[mask&(full['crop']==c)])) for c in range(5)]
  quality_s=quality[mask];rec=dict(subject=str(s),diagnosis='DEP' if full['diagnosis'][mask][0]==1 else 'HC',auc_40windows=auc,mean_samecrop8_auc=float(np.mean(cropauc)),auc_crop0=cropauc[0],auc_crop4=cropauc[-1],mean_channel_std=float(np.median(quality_s[:,0])),max_abs=float(quality_s[:,1].max()),median_clipped_fraction=float(np.median(quality_s[:,2])),mean_clipped_fraction=float(quality_s[:,2].mean()),spike_ratio=float(np.max(quality_s[:,1])/max(np.median(quality_s[:,0]),1e-10)),source_context_distance=dist.get(str(s)),score_std=float(q[mask].std()),positive_negative_mean_gap=float(q[mask&(y==1)].mean()-q[mask&(y==0)].mean()),flat_signal=bool(np.max(quality_s[:,0])<.01))
  if len(ix):
   # Hold out an entire positive and neutral trial, all5crops excluded.
   # Each target trial gets4heldout predictions, averaged. This diagnostic
   # uses labels from6other trials of the same person, unlike competition.
   pp=np.zeros(len(ix));counts=np.zeros(len(ix));trials=d['trial'][ix];yy=d['label'][ix]
   for neg in range(1,5):
    for pos in range(5,9):
     va=(trials==neg)|(trials==pos);tr=~va;sc=StandardScaler().fit(xx[ix[tr]]);model=LogisticRegression(C=.1,solver='liblinear',max_iter=500,random_state=1031).fit(sc.transform(xx[ix[tr]]),yy[tr]);pp[va]+=model.predict_proba(sc.transform(xx[ix[va]]))[:,1];counts[va]+=1
   assert np.all(counts==4);pp/=counts;probe[mapping[ix]]=pp;rec['within_person_labeled_probe_auc']=float(roc_auc_score(yy,pp));rec['temporal_DE_variability']=float(np.median(d['sequence'][ix].std(1)))
  else:rec['within_person_labeled_probe_auc']=None;rec['temporal_DE_variability']=None
  records.append(rec)
 print('PERSON AUDIT COMPLETE',len(records),flush=True)
 associations={}
 for name in ['HC','DEP']:
  rr=[r for r in records if r['diagnosis']==name and not r['flat_signal']];err=[1-r['auc_40windows'] for r in rr];associations[name]={}
  for key in ['mean_channel_std','mean_clipped_fraction','spike_ratio','source_context_distance','temporal_DE_variability']:
   associations[name][key]=rho_ci([r[key] for r in rr],err)
 # Length and time-position checks keep the same frozen10s forecasts.
 length={};cross={};probe_summary={}
 for name,mask in [('all',np.ones(len(y),dtype=bool)),('HC',full['diagnosis']==0),('DEP',full['diagnosis']==1)]:
  crop_scores=[float(roc_auc_score(y[mask&(full['crop']==c)],q[mask&(full['crop']==c)])) for c in range(5)]
  yy=[];qq=[]
  for s in np.unique(g[mask]):
   for t in range(1,9):
    z=(g==s)&(full['trial']==t);yy.append(int(y[z][0]));qq.append(float(q[z].mean()))
  length[name]=dict(crop0_to4_auc=crop_scores,pooled10second_auc=float(roc_auc_score(y[mask],q[mask])),averaged50second_trial_auc_DIAGNOSTIC_ONLY=float(roc_auc_score(yy,qq)),not_a_new_model=True,not_official_metric=True)
  rr=[r for r in records if (name=='all' or r['diagnosis']==name) and not r['flat_signal']];probe_summary[name]=dict(n_people=len(rr),mean_cross_subject_within_person_auc=float(np.mean([r['auc_40windows'] for r in rr])),mean_person_labeled_probe_auc=float(np.mean([r['within_person_labeled_probe_auc'] for r in rr])),people_probe_better=sum(r['within_person_labeled_probe_auc']>r['auc_40windows'] for r in rr))
  pos=q[mask&(y==1)];neg=q[mask&(y==0)];sp=g[mask&(y==1)];sn=g[mask&(y==0)];ok=(pos[:,None]>neg[None]).astype(float)+.5*(pos[:,None]==neg[None]);same=sp[:,None]==sn[None];cross[name]=dict(within_person_pair_auc=float(ok[same].mean()),between_person_pair_auc=float(ok[~same].mean()))
 out=dict(descriptive_only=True,official_public_error_labels_not_available=True,no_official_public_signals_read=True,no_model_replacement=True,person_records=records,associations=associations,length=length,pair_decomposition=cross,within_person_probe=probe_summary,probe_limitations=['Different task: uses6labeled trials from target person; not deployable to unlabeled public test','All crops of heldout trials excluded from supervised fitting','Source labels are condition/block labels; probe cannot distinguish emotion from recording-condition confounds','Overlapping repeated CV predictions are not independent evidence'],causal_limitations='Observational associations and diagnostic probes do not identify physiological causes; no causal claim',official_baseline_user_report=88.13)
 driver.save(OUT/'audit.json',out)
 with (OUT/'per_person.csv').open('w',encoding='utf-8-sig',newline='') as h:
  w=csv.DictWriter(h,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 np.savez_compressed(OUT/'diagnostic_probe_predictions_NOT_SUBMISSION.npz',subject=g,probability=probe)
 print(json.dumps({k:out[k] for k in ['associations','length','pair_decomposition','within_person_probe']}),flush=True)
if __name__=='__main__':main()
