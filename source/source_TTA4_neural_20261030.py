"""Four fixed signal views, development-only held-subject checkpoint inference."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import json,traceback
import numpy as np,torch
import driver
from leaderboard_dualbranch_20261028 import Inputs,Net,groups
from select_ensemble import postprocess
OUT=driver.JOB/'source_TTA4_neural_20261030';OUT.mkdir(exist_ok=True)
MODES=['de_cnn','raw_de_cnn','raw_de_bigru']
def main():
 torch.set_num_threads(1);p=driver.protocol();spec=dict(views=['identity','reverse','roll5seconds','reverse_then_roll5seconds'],augmentation_fixed_before_current_outer_results=True,source_checkpoints_only=True,query_labels_not_used=True,official_data_not_opened=True,seed_unchanged=True,script_sha256=driver.sha(__file__))
 if (OUT/'protocol.json').exists():assert json.loads((OUT/'protocol.json').read_text())==spec
 else:driver.save(OUT/'protocol.json',spec)
 inputs=Inputs();inputs.d['label'][:]=-1;inputs.d['diagnosis'][:]=-1;audits=[]
 for f in p['folds']:
  for part in ['inner0','inner1','inner2','outer']:
   for mode in MODES:
    name=f'{mode}_fold{f["fold"]}_{part}';dest=OUT/(name+'.npz');src=driver.JOB/'leaderboard_dualbranch_20261028'/(name+'.npz');modelpath=src.with_suffix('.pt')
    if dest.exists():
     with np.load(dest) as a:assert str(a['checkpoint_sha256'])==driver.sha(modelpath);error=float(a['identity_error'])
    else:
     with np.load(src) as a:va=a['validation'];reference=a['probability'];assert str(a['model_sha256'])==driver.sha(modelpath)
     saved=torch.load(modelpath,map_location='cpu',weights_only=True);model=Net(mode);u,rr=groups(inputs.d,va);views=[]
     with torch.inference_mode():
      for variant in range(4):
       states=[]
       for state in saved['states']:
        model.load_state_dict(state);model.eval();pred=[]
        for j in range(0,len(rr),4):
         de,raw=inputs.get(rr[j:j+4],mode)
         if variant in (1,3):de=de.flip(2);raw=raw.flip(-1) if mode!='de_cnn' else raw
         if variant in (2,3):de=torch.roll(de,5,2);raw=torch.roll(raw,625,-1) if mode!='de_cnn' else raw
         pred.append(model(de,raw).numpy())
        states.append(np.concatenate(pred))
       views.append(np.mean(states,axis=0))
     grouped=np.repeat(u,8);identity=postprocess(views[0].ravel(),grouped,'conditional',1.);averaged=postprocess(np.mean(views,axis=0).ravel(),grouped,'conditional',1.);lookup={v:j for j,v in enumerate(va)};positions=np.array([lookup[v] for v in rr.ravel()]);q0=np.empty(len(va));q=np.empty(len(va));q0[positions]=identity;q[positions]=averaged;error=float(abs(q0-reference).max());assert error<1e-5,(name,error)
     np.savez_compressed(dest,validation=va,probability=q,identity_probability=q0,identity_error=error,checkpoint_sha256=np.array(driver.sha(modelpath)))
    audits.append(dict(task=name,identity_replay_error=error));driver.save(OUT/'progress.json',dict(completed=len(audits),total=60));print('SOURCE TTA',len(audits),'/60',name,flush=True)
 driver.save(OUT/'verification.json',dict(checks_passed=True,all60_tasks_complete=True,maximum_identity_delta=max(a['identity_replay_error'] for a in audits),audits=audits,official_and_sealed_not_opened=True,no_parameter_updates_on_queries=True))
if __name__=='__main__':
 try:main()
 except BaseException:driver.save(OUT/'failure.json',dict(traceback=traceback.format_exc()));raise
