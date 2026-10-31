"""Compute independent fold5 inner tasks ahead on the second CPU process.
Same fit function/hyperparameters. Stop publishing when main reaches fold4,
leaving an entire fold of separation; never select models or touch outer labels.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import json,time
import numpy as np,torch
import dynamic_connectivity_20261101 as m

def main():
 torch.set_num_threads(1);data=m.Data();d=data.d;fold=m.driver.protocol()['folds'][4];staging=m.OUT/'parallel_staging';staging.mkdir(exist_ok=True);records=[]
 def main_fold():return json.loads((m.OUT/'progress.json').read_text()).get('fold',99)
 for part in range(3):
  split=fold['inner'][part];tr=np.flatnonzero(np.isin(d['subject'],split['training_subjects'])&data.eligible);va=np.flatnonzero(np.isin(d['subject'],split['validation_subjects']))
  for mode in m.MODES:
   if main_fold()>=4:m.driver.save(m.OUT/'parallel_worker.json',dict(records=records,stopped_for_separation=True));return
   final=m.OUT/f'{mode}_fold5_inner{part}.pt';cache=final.with_suffix('.npz')
   if cache.exists():continue
   temp=staging/final.name;start=time.time();q,error=m.fit(data,tr,va,mode,temp)
   if main_fold()>=4:m.driver.save(m.OUT/'parallel_worker.json',dict(records=records,stopped_for_separation=True,unpublished_checkpoint=str(temp)));return
   if cache.exists():continue
   temp.replace(final);payload=staging/cache.name;np.savez_compressed(payload,validation=va,probability=q,reload_error=error,model_sha256=np.array(m.driver.sha(final)));payload.replace(cache);records.append(dict(fold=5,part=part,mode=mode,seconds=time.time()-start,checkpoint_sha256=m.driver.sha(final),reload_error=error));m.driver.save(m.OUT/'parallel_worker.json',dict(records=records,fit_function_sha256=m.driver.sha(m.__file__),hyperparameters_unchanged=True));print('PARALLEL FOLD5',part,mode,'DONE',flush=True)
 m.driver.save(m.OUT/'parallel_worker.json',dict(records=records,complete=True,fit_function_sha256=m.driver.sha(m.__file__),hyperparameters_unchanged=True))
if __name__=='__main__':main()
