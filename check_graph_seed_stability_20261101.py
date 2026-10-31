"""Reload five graph-seed fusion replicas through the existing native checker.
Cached10s EEG features feed the fixed graph; old83 raw models aren't replayed.
"""
import os
from pathlib import Path
import json,subprocess,sys
import driver
OUT=driver.JOB/'graph_seed_stability_20261101'

def main():
 spec=json.loads((OUT/'protocol.json').read_text());source=(driver.JOB/'check_connectivity_graph_20261101.py').read_text();checks=[]
 for seed in spec['seeds']:
  generated=source.replace("fusion=driver.JOB/'connectivity_graph_fusion_20261101'",f"fusion=driver.JOB/'graph_seed_stability_20261101/seed{seed}'");assert generated!=source;script=OUT/f'check_seed{seed}.py';script.write_text(generated)
  with (OUT/f'check_seed{seed}.log').open('w') as log:subprocess.run([sys.executable,str(script)],cwd=driver.JOB,env={**os.environ,'PYTHONPATH':str(driver.JOB)},stdout=log,stderr=subprocess.STDOUT,check=True)
  a=json.loads((OUT/f'seed{seed}/independent_reload.json').read_text());assert max(a['cached_EEG_feature_native_CNN_errors'])<1e-6;assert max(a['native_fusion_errors'])<1e-6;checks.append(dict(seed=seed,graph_errors=a['cached_EEG_feature_native_CNN_errors'],fusion_errors=a['native_fusion_errors'],checker_sha256=driver.sha(script)));print('GRAPH SEED REPLAY',seed,'PASSED',flush=True)
 driver.save(OUT/'independent_seed_replay.json',dict(checks=checks,all25fold_models_passed=True,total_fusion_heads=400,fixed_graph_cached_EEG_replayed=True,old83_raw_MAT_models_replayed=False,public_data_used=False))
if __name__=='__main__':main()
