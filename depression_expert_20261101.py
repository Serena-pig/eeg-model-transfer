"""Source-only clinical expert adaptation of the consistency CNN.
Head-only vs head+temporal layer; healthy replay; source3inner selection.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import json,copy,random,time,shutil
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
import driver
from leaderboard_dualbranch_20261028 import Inputs,Net,groups,infer
from select_ensemble import ASSIGNMENTS,postprocess
OUT=driver.JOB/'depression_expert_20261101';OUT.mkdir(exist_ok=True)
STRATEGIES=['head','head_time'];EPOCHS=12;SEED=110103

def fit(inputs,tr,va,parent,strategy,path):
 torch.set_num_threads(1);torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED);rng=np.random.default_rng(SEED);d=inputs.d;model=Net('de_cnn');warm=torch.load(parent,map_location='cpu',weights_only=True);assert set(warm['source_subjects'])==set(np.unique(d['subject'][tr]).tolist());model.load_state_dict(warm['states'][-1]);reference={k:v.detach().clone() for k,v in model.named_parameters()}
 for name,param in model.named_parameters():param.requires_grad_(name.startswith('head.') or (strategy=='head_time' and name.startswith('de_time.')))
 parameters=[p for p in model.parameters() if p.requires_grad];optimizer=torch.optim.AdamW(parameters,lr=.0003,weight_decay=.03);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,EPOCHS,eta_min=.00003);people=np.unique(d['subject'][tr]);rows=[];diag=[]
 for s in people:
  ix=tr[d['subject'][tr]==s];rows.append(np.stack([ix[d['trial'][ix]==t][np.argsort(d['crop'][ix[d['trial'][ix]==t]])] for t in range(1,9)]));diag.append(d['diagnosis'][ix[0]])
 rows=np.array(rows);diag=np.array(diag);pools=[np.flatnonzero(diag==v) for v in [0,1]];steps=int(np.ceil(len(pools[1])*5/4));assign=torch.tensor(ASSIGNMENTS.T,dtype=torch.float32);states=[]
 for epoch in range(1,EPOCHS+1):
  model.train()
  for step in range(steps):
   who=np.array([rng.choice(pools[0])]+[rng.choice(pools[1]) for _ in range(3)]);crops=rng.integers(0,5,(4,8));rr=rows[who[:,None],np.arange(8)[None],crops];rr2=rows[who[:,None],np.arange(8)[None],(crops+rng.integers(1,5,(4,8)))%5];a,b=inputs.get(rr,'de_cnn');a2,b2=inputs.get(rr2,'de_cnn');z=model(a+torch.randn_like(a)*.02,b);z2=model(a2,b2);y=torch.tensor(d['label'][rr],dtype=torch.float32);flat=z.flatten();yy=y.flatten();pair=F.softplus(flat[yy<.5][None]-flat[yy>.5][:,None]).mean();cond=(torch.logsumexp(z@assign,1)-(z*y).sum(1)).mean()/4;anchor=sum((p-reference[n]).square().sum() for n,p in model.named_parameters() if p.requires_grad);loss=.5*F.binary_cross_entropy_with_logits(z,y*.98+.01)+.25*pair+.25*cond+.002*z.square().mean()+F.mse_loss(z,z2)+.01*anchor;optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(parameters,3);optimizer.step()
  scheduler.step()
  if epoch in [8,10,12]:states.append(copy.deepcopy(model.state_dict()))
 u,rr=groups(d,va);pred=[]
 for state in states:model.load_state_dict(state);pred.append(infer(model,inputs,rr))
 z=np.mean(pred,axis=0);q=postprocess(z.ravel(),np.repeat(u,8),'conditional',1.);out=np.zeros(len(va));lookup={v:j for j,v in enumerate(va)};out[[lookup[v] for v in rr.ravel()]]=q
 torch.save(dict(states=states,mode='de_cnn',strategy=strategy,epochs=EPOCHS,source_subjects=people.tolist(),parent_sha256=driver.sha(parent)),path);saved=torch.load(path,map_location='cpu',weights_only=True);replay=[]
 for state in saved['states']:model.load_state_dict(state);replay.append(infer(model,inputs,rr[:,::-1]))
 error=float(abs(np.mean(replay,0)[:,::-1]-z).max());assert error<1e-5
 return out,error

def main():
 torch.set_num_threads(1);spec=dict(strategies=STRATEGIES,epochs=12,seed=SEED,source_sampling='3DEP+1HC healthy replay',lr=.0003,weight_decay=.03,l2_anchor=.01,consistency_weight=1.,selection='Highest source-inner DEP AUC, pooled source AUC floor warmup-.005; require DEP gain .005, else warmup',primary_final_metric='unweighted pooled AUC on unchanged2400records',no_outer_selection=True,public_test_used=False,holdout_used=False,script_sha256=driver.sha(__file__))
 p=OUT/'protocol.json'
 if p.exists():assert json.loads(p.read_text())==spec
 else:driver.save(p,spec)
 inputs=Inputs();d=inputs.d;selections=[];completed=0
 for fold in driver.protocol()['folds']:
  k=fold['fold'];source_predictions={s:[] for s in ['warmup']+STRATEGIES};source_y=[];source_diag=[]
  for part in range(3):
   split=fold['inner'][part];tr=np.flatnonzero(np.isin(d['subject'],split['training_subjects'])&inputs.eligible);va=np.flatnonzero(np.isin(d['subject'],split['validation_subjects']));assert not set(d['subject'][tr])&set(d['subject'][va]);parent=driver.JOB/f'neural_trial_consistency_20261031/gamma1.0_fold{k}_inner{part}.pt'
   with np.load(parent.with_suffix('.npz')) as a:assert np.array_equal(a['validation'],va);source_predictions['warmup'].append(a['probability'])
   source_y.append(d['label'][va]);source_diag.append(d['diagnosis'][va])
   for strategy in STRATEGIES:
    path=OUT/f'{strategy}_fold{k}_inner{part}.pt';cache=path.with_suffix('.npz')
    if cache.exists():
     with np.load(cache) as a:q=a['probability'];assert str(a['model_sha256'])==driver.sha(path)
    else:
     started=time.time();driver.save(OUT/'progress.json',dict(stage='training',completed_inner_fits=completed,total_inner_fits=30,fold=k,part=part,strategy=strategy));q,error=fit(inputs,tr,va,parent,strategy,path);np.savez_compressed(cache,validation=va,probability=q,reload_error=error,model_sha256=np.array(driver.sha(path)));print('CLINICAL EXPERT',k,part,strategy,'seconds',round(time.time()-started,1),flush=True)
    source_predictions[strategy].append(q);completed+=1
  yy=np.concatenate(source_y);dd=np.concatenate(source_diag);metrics={s:dict(pooled_auc=float(roc_auc_score(yy,np.concatenate(q))),DEP_auc=float(roc_auc_score(yy[dd==1],np.concatenate(q)[dd==1]))) for s,q in source_predictions.items()};eligible=[s for s in STRATEGIES if metrics[s]['pooled_auc']>=metrics['warmup']['pooled_auc']-.005 and metrics[s]['DEP_auc']>=metrics['warmup']['DEP_auc']+.005];chosen=max(eligible,key=lambda s:metrics[s]['DEP_auc']) if eligible else 'warmup';selection=dict(fold=k,chosen=chosen,source_metrics=metrics);selections.append(selection);driver.save(OUT/f'fold{k}_selection.json',selection)
  for part in range(3):
   src=driver.JOB/f'neural_trial_consistency_20261031/gamma1.0_fold{k}_inner{part}.npz' if chosen=='warmup' else OUT/f'{chosen}_fold{k}_inner{part}.npz';shutil.copy2(src,OUT/f'selected_fold{k}_inner{part}.npz')
  va=np.flatnonzero(np.isin(d['subject'],fold['validation_subjects']));cache=OUT/f'selected_fold{k}_outer.npz'
  if chosen=='warmup':shutil.copy2(driver.JOB/f'neural_trial_consistency_20261031/selected_fold{k}_outer.npz',cache)
  elif not cache.exists():
   tr=np.flatnonzero(np.isin(d['subject'],fold['training_subjects'])&inputs.eligible);parent=driver.JOB/f'neural_trial_consistency_20261031/selected_fold{k}_outer.pt';path=OUT/f'selected_fold{k}_outer.pt';q,error=fit(inputs,tr,va,parent,chosen,path);np.savez_compressed(cache,validation=va,probability=q,reload_error=error,model_sha256=np.array(driver.sha(path)))
  print('CLINICAL EXPERT OUTER',k,'/5 COMPLETE',chosen,flush=True)
 # Full diagnostic scoring only after all fixed source decisions are saved.
 out=np.full(len(d['label']),.5)
 for k in range(1,6):
  with np.load(OUT/f'selected_fold{k}_outer.npz') as a:out[a['validation']]=a['probability']
 dep=d['diagnosis']==1;driver.save(OUT/'summary.json',dict(selections=selections,primary_pooled_auc=float(roc_auc_score(d['label'],out)),DEP_auc=float(roc_auc_score(d['label'][dep],out[dep])),all5outer3inner_complete=True,cohort_reused=True,public_data_used=False,official_submission=False));np.savez_compressed(OUT/'selected_oof.npz',probability=out);print('CLINICAL EXPERT ALL COMPLETE',flush=True)
if __name__=='__main__':main()
