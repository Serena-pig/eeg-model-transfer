"""Five matched fusion-seed checks of frozen correlation-graph augmentation.
Baseline controls are the already-trained and verified same-seed controls.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import sys,json,subprocess
import numpy as np
from sklearn.metrics import roc_auc_score
import driver
from postflight import pair_matrix
OUT=driver.JOB/'graph_seed_stability_20261101';OUT.mkdir(exist_ok=True)
SEEDS=[1031,1032,1033,1034,1035]

def main():
 controls={str(seed):driver.JOB/f'seed_stability_20261101/seed{seed}/baseline/subbag16_rawmean_calibrated_oof.npz' for seed in SEEDS};frozen_manifest=json.loads((driver.JOB/'frozen_baselines/seed_stability_20261101/SHA256.json').read_text())
 for seed in SEEDS:assert driver.sha(controls[str(seed)])==frozen_manifest[f'seed{seed}/baseline/subbag16_rawmean_calibrated_oof.npz']
 graph_files=sorted((driver.JOB/'connectivity_graph_20261101').glob('selected_fold*.npz'))+sorted((driver.JOB/'connectivity_graph_20261101').glob('selected_fold*.pt'))
 spec=dict(seeds=SEEDS,reference_seed=1030,graph_training_seed_frozen=110131,scope='Fusion patient-subbag seed only, not whole graph/base model retraining',primary_metric='unweighted pooled AUC on unchanged60people2400records',comparison='Each new graph fusion versus corresponding same-seed87member baseline',baseline_controls_reused=True,base_and_graph_weights_frozen=True,full_subject_splits_unchanged=True,new_fusion_head_fits=400,seed_selection=False,criterion='At least4/5paired gains positive AND mean positive AND lower paired-person CI above0; still reused-cohort evidence',baseline_sha256={k:driver.sha(v) for k,v in controls.items()},graph_input_sha256={str(p.relative_to(driver.JOB)):driver.sha(p) for p in graph_files},public_data_used=False,holdout_used=False,official_submissions=0,script_sha256=driver.sha(__file__))
 if (OUT/'protocol.json').exists():assert json.loads((OUT/'protocol.json').read_text())==spec
 else:driver.save(OUT/'protocol.json',spec)
 audit=[]
 for seed in SEEDS:
  source=(driver.JOB/'connectivity_graph_fusion_20261101.py').read_text();generated=source.replace("OUT=driver.JOB/'connectivity_graph_fusion_20261101'",f"OUT=driver.JOB/'graph_seed_stability_20261101/seed{seed}'").replace('seed_fixed=1030',f'seed_fixed={seed}').replace("1030+f['fold']*100",f"{seed}+f['fold']*100").replace('CPU_workers=1','CPU_workers=2').replace('ThreadPoolExecutor(max_workers=1)','ThreadPoolExecutor(max_workers=2)');script=OUT/f'graph_fusion_seed{seed}.py';script.write_text(generated)
  reverse=generated.replace(f"OUT=driver.JOB/'graph_seed_stability_20261101/seed{seed}'","OUT=driver.JOB/'connectivity_graph_fusion_20261101'").replace(f'seed_fixed={seed}','seed_fixed=1030').replace(f"{seed}+f['fold']*100","1030+f['fold']*100").replace('CPU_workers=2','CPU_workers=1').replace('ThreadPoolExecutor(max_workers=2)','ThreadPoolExecutor(max_workers=1)');assert reverse==source;audit.append(dict(seed=seed,only_seed_paths_and_worker_count_changed=True,generated_script_sha256=driver.sha(script)))
  if not (OUT/f'seed{seed}/summary.json').exists():
   driver.save(OUT/'progress.json',dict(stage='training',seed=seed,completed_seeds=SEEDS.index(seed),total_seeds=5))
   with (OUT/f'seed{seed}.log').open('w') as log:subprocess.run([sys.executable,'-u',str(script)],cwd=driver.JOB,env={**os.environ,'PYTHONPATH':str(driver.JOB)},stdout=log,stderr=subprocess.STDOUT,check=True)
  print('MATCHED GRAPH SEED COMPLETE',seed,flush=True)
 driver.save(OUT/'code_change_audit.json',audit)
 with np.load(driver.CONTROL/'full_evaluation_rows.npz') as a:y=a['label'];subjects=a['subject'];diagnosis=a['diagnosis']
 people=np.unique(subjects);matrices=[];predictions=[];per_seed=[]
 for seed in SEEDS:
  with np.load(controls[str(seed)]) as a:old=a['probability']
  with np.load(OUT/f'seed{seed}/subbag16_rawmean_calibrated_oof.npz') as a:new=a['probability']
  before=float(roc_auc_score(y,old));after=float(roc_auc_score(y,new));per_seed.append(dict(seed=seed,baseline_auc=before,graph_fusion_auc=after,paired_delta=after-before,baseline_DEP_auc=float(roc_auc_score(y[diagnosis==1],old[diagnosis==1])),graph_DEP_auc=float(roc_auc_score(y[diagnosis==1],new[diagnosis==1]))));matrices.append(pair_matrix(y,new,subjects,people)-pair_matrix(y,old,subjects,people));predictions.append((old,new))
 mean_matrix=np.mean(matrices,0);rng=np.random.default_rng(1101);counts=np.zeros((10000,len(people)))
 for j in range(len(counts)):
  for name in ['HC','DEP']:
   ix=np.flatnonzero(np.char.startswith(people,name));counts[j]+=np.bincount(rng.choice(ix,len(ix),replace=True),minlength=len(people))
 boot=np.einsum('bi,ij,bj->b',counts,mean_matrix,counts,optimize=True)/counts.sum(1)**2;ci=np.quantile(boot,[.025,.975]);loo=[]
 for s in people:
  mask=subjects!=s;loo.append(dict(subject=str(s),mean_delta=float(np.mean([roc_auc_score(y[mask],new[mask])-roc_auc_score(y[mask],old[mask]) for old,new in predictions]))))
 influential=sorted(loo,key=lambda v:v['mean_delta'])[:2];mask=~np.isin(subjects,[v['subject'] for v in influential]);remove2=float(np.mean([roc_auc_score(y[mask],new[mask])-roc_auc_score(y[mask],old[mask]) for old,new in predictions]));positive=sum(r['paired_delta']>0 for r in per_seed);mean=float(np.mean([r['paired_delta'] for r in per_seed]));passed=bool(positive>=4 and mean>0 and ci[0]>0)
 for k,p in controls.items():assert driver.sha(p)==spec['baseline_sha256'][k]
 for k,sha in spec['graph_input_sha256'].items():assert driver.sha(driver.JOB/k)==sha
 result=dict(primary_metric=spec['primary_metric'],per_seed=per_seed,positive_seeds=positive,total_new_seeds=5,mean_paired_auc_delta=mean,paired_person_bootstrap10000_95CI=ci.tolist(),leave_one_person_out=loo,leave_one_person_out_positive_count=sum(r['mean_delta']>0 for r in loo),descriptive_remove2=[r['subject'] for r in influential],descriptive_delta_remove2=remove2,predeclared_criterion_passed=passed,graph_weight_randomness_tested=False,only_fusion_seed_varied=True,cohort_reused=True,independent_generalization_confirmed=False,promotion=False,holdout_used=False,official_submissions=0)
 driver.save(OUT/'summary.json',result);driver.save(OUT/'progress.json',dict(stage='complete',completed_seeds=5,total_seeds=5));print('ALL GRAPH SEEDS COMPLETE',json.dumps({k:v for k,v in result.items() if k!='leave_one_person_out'}),flush=True)
if __name__=='__main__':main()
