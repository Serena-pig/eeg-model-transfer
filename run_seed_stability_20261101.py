"""Five prespecified new fusion seeds, matched residual/control comparison.
Original base/CNN models, subject splits and unweighted AUC stay fixed.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json,hashlib,subprocess,sys
import numpy as np
from sklearn.metrics import roc_auc_score
import driver
from postflight import pair_matrix
OUT=driver.JOB/'seed_stability_20261101';OUT.mkdir(exist_ok=True)
SEEDS=[1031,1032,1033,1034,1035]

def generate(seed):
 # Generated scripts are isolated copies; no original/frozen file is edited.
 source=(driver.JOB/'pooled_rank_bag_20261030.py').read_text()
 source=source.replace("OUT=driver.JOB/'pooled_rank_bag_20261030'",f"OUT=driver.JOB/'seed_stability_20261101/seed{seed}/baseline'")
 source=source.replace('seed_fixed=1030',f'seed_fixed={seed}').replace("1030+f['fold']*100",f"{seed}+f['fold']*100")
 source=source.replace("spec=dict(variants=NAMES,","spec=dict(primary_metric='unweighted_pooled_auc',stability_only=True,no_best_seed_selection=True,variants=NAMES,")
 baseline=OUT/f'baseline_seed{seed}.py';baseline.write_text(source)
 source=(driver.JOB/'consistency_residual_20261101.py').read_text()
 source=source.replace("OUT=driver.JOB/'consistency_residual_20261101'",f"OUT=driver.JOB/'seed_stability_20261101/seed{seed}/residual'")
 source=source.replace("'pooled_rank_bag_20261030/",f"'seed_stability_20261101/seed{seed}/baseline/")
 residual=OUT/f'residual_seed{seed}.py';residual.write_text(source)
 return baseline,residual

def main():
 parent=driver.JOB/'consistency_residual_20261101';original=driver.JOB/'pooled_rank_bag_20261030/subbag16_rawmean_calibrated_oof.npz';book=Path('/workspace/rank16_release/EEG_RANK16_20261030/submission_rank16.xlsx')
 spec=dict(seeds=SEEDS,reference_seed=1030,seed_variation='Patient subbag draws in original16 fusion heads only; all old/CNN weights frozen',primary_metric='unweighted pooled ROC AUC on unchanged2400development records',full_validation='same5outer3inner subject splits',comparison='400total baseline heads plus25total2parameter residual heads across5new seeds',residual_hyperparameters_fixed=True,best_seed_selection=False,clinical_reweighting=False,holdout_used=False,public_test_used=False,submissions=0,criterion='At least4/5positive paired seed deltas, mean delta positive, AND lower person-cluster CI above0; still reused-cohort evidence only',original_prediction_sha256=driver.sha(original),baseline_workbook_sha256=driver.sha(book),script_sha256=driver.sha(__file__))
 protocol=OUT/'protocol.json'
 if protocol.exists():assert json.loads(protocol.read_text())==spec
 else:driver.save(protocol,spec)
 for seed in SEEDS:
  baseline,residual=generate(seed);root=OUT/f'seed{seed}';root.mkdir(exist_ok=True)
  for stage,script,done in [('baseline',baseline,root/'baseline/summary.json'),('residual',residual,root/'residual/summary.json')]:
   if done.exists():continue
   driver.save(OUT/'progress.json',dict(stage=stage,seed=seed,completed_seeds=SEEDS.index(seed),total_seeds=5))
   with (root/(stage+'.log')).open('w') as log:
    subprocess.run([sys.executable,'-u',str(script)],cwd=driver.JOB,env={**os.environ,'PYTHONPATH':str(driver.JOB)},stdout=log,stderr=subprocess.STDOUT,check=True)
  print('MATCHED SEED COMPLETE',seed,flush=True)
 # No seed selection or pooled outer scoring until every training task finishes.
 with np.load(driver.CONTROL/'full_evaluation_rows.npz') as a:y=a['label'];subjects=a['subject'];diagnosis=a['diagnosis']
 people=np.unique(subjects);matrices=[];per_seed=[];predictions=[]
 for seed in SEEDS:
  root=OUT/f'seed{seed}'
  with np.load(root/'baseline/subbag16_rawmean_calibrated_oof.npz') as a:old=a['probability']
  with np.load(root/'residual/oof.npz') as a:new=a['probability']
  baseauc=float(roc_auc_score(y,old));newauc=float(roc_auc_score(y,new));per_seed.append(dict(seed=seed,baseline_auc=baseauc,residual_auc=newauc,delta=newauc-baseauc,baseline_DEP_auc=float(roc_auc_score(y[diagnosis==1],old[diagnosis==1])),residual_DEP_auc=float(roc_auc_score(y[diagnosis==1],new[diagnosis==1]))));matrices.append(pair_matrix(y,new,subjects,people)-pair_matrix(y,old,subjects,people));predictions.append((old,new))
 delta=np.mean(matrices,0);rng=np.random.default_rng(1101);counts=np.zeros((10000,len(people)))
 for j in range(len(counts)):
  for name in ['HC','DEP']:
   ix=np.flatnonzero(np.char.startswith(people,name));counts[j]+=np.bincount(rng.choice(ix,len(ix),replace=True),minlength=len(people))
 boot=np.einsum('bi,ij,bj->b',counts,delta,counts,optimize=True)/counts.sum(1)**2;interval=np.quantile(boot,[.025,.975]);loo=[]
 for s in people:
  ix=subjects!=s;value=float(np.mean([roc_auc_score(y[ix],new[ix])-roc_auc_score(y[ix],old[ix]) for old,new in predictions]));loo.append(dict(subject=str(s),mean_seed_delta=value))
 # Descriptive influence check: removing2people selected AFTER analysis is not
 # a new validation score or a rule for removing subjects from the benchmark.
 influential=sorted(loo,key=lambda a:a['mean_seed_delta'])[:2];ix=~np.isin(subjects,[a['subject'] for a in influential]);remove2=float(np.mean([roc_auc_score(y[ix],new[ix])-roc_auc_score(y[ix],old[ix]) for old,new in predictions]));positive=sum(a['delta']>0 for a in per_seed);mean=float(np.mean([a['delta'] for a in per_seed]));passed=bool(positive>=4 and mean>0 and interval[0]>0)
 assert driver.sha(original)==spec['original_prediction_sha256'];assert driver.sha(book)==spec['baseline_workbook_sha256']
 summary=dict(primary_metric=spec['primary_metric'],per_seed=per_seed,positive_seeds=positive,total_new_seeds=5,mean_paired_auc_delta=mean,paired_person_10000bootstrap95CI=interval.tolist(),leave_one_person_out=loo,leave_one_person_out_positive_count=sum(a['mean_seed_delta']>0 for a in loo),descriptive_remove2most_influential=[a['subject'] for a in influential],descriptive_delta_after_removing2=remove2,predeclared_stability_criterion_passed=passed,cohort_reused=True,independent_generalization_proven=False,only_fusion_seed_varied=True,baseline_unchanged=True,holdout_used=False,official_submission=False)
 driver.save(OUT/'summary.json',summary);driver.save(OUT/'progress.json',dict(stage='complete',completed_seeds=5,total_seeds=5));print('ALL SEED STABILITY COMPLETE',json.dumps({k:v for k,v in summary.items() if k!='leave_one_person_out'}),flush=True)
if __name__=='__main__':main()
