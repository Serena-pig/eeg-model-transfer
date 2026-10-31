"""Label-free source-development125Hz signal -> matched static/local graphs.
Filter whole10s first; split into five2s blocks; fixed10% diagonal shrinkage.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import json,time
from pathlib import Path
import numpy as np
from scipy.signal import butter,sosfiltfilt
from threadpoolctl import threadpool_limits
import driver
OUT=driver.JOB/'local_connectivity_cache_20261101';OUT.mkdir(exist_ok=True)
BANDS=[(1,4),(4,8),(8,13),(13,20),(20,30),(30,45),(1,45)]

def correlation(x):
 x=x-x.mean(-1,keepdims=True);c=x@x.swapaxes(-1,-2)/x.shape[-1];trace=np.trace(c,axis1=-2,axis2=-1)/30;c=.9*c+.1*trace[...,None,None]*np.eye(30);sd=np.sqrt(np.maximum(np.diagonal(c,axis1=-2,axis2=-1),1e-12));return np.clip(c/sd[...,None]/sd[...,None,:],-1,1)

def main():
 threadpool_limits(1)
 if (OUT/'completion.json').exists():print('LOCAL CONNECTIVITY CACHE EXISTS',flush=True);return
 raw=np.load(driver.ART/'raw_125hz.npy',mmap_mode='r')
 with np.load(driver.ART/'features.npz') as a:subjects=a['subject'];assert a['train'].all();assert set(np.unique(subjects)).isdisjoint(driver.protocol()['split']['holdout'])
 assert raw.shape==(2360,30,1250);spec=dict(fs=125,input_seconds=10,local_seconds=2,local_blocks=5,bands=BANDS,shrinkage=.1,stored_dtype='float16',stored_precision_max_abs_error_bound=.00025,filter='4thorderButterworth zero-phase on whole10s before splitting',raw_input_sha256=driver.sha(driver.ART/'raw_125hz.npy'),subject_array_sha256=__import__('hashlib').sha256(subjects.tobytes()).hexdigest(),public_data_used=False,holdout_used=False,labels_used_for_features=False,script_sha256=driver.sha(__file__));driver.save(OUT/'protocol.json',spec)
 static=np.lib.format.open_memmap(OUT/'static_corr.npy',mode='w+',dtype='float16',shape=(len(raw),7,30,30));dynamic=np.lib.format.open_memmap(OUT/'local_corr.npy',mode='w+',dtype='float16',shape=(len(raw),5,7,30,30));filters=[butter(4,b,fs=125,btype='bandpass',output='sos') for b in BANDS];quantization=0.
 for start in range(0,len(raw),16):
  stop=min(start+16,len(raw));x=np.asarray(raw[start:stop],dtype='float64')
  for band,sos in enumerate(filters):
   filtered=sosfiltfilt(sos,x,axis=-1);s=correlation(filtered);local=filtered.reshape(len(x),30,5,250).transpose(0,2,1,3);c=correlation(local);static[start:stop,band]=s;dynamic[start:stop,:,band]=c;quantization=max(quantization,float(abs(s-s.astype('float16').astype(float)).max()),float(abs(c-c.astype('float16').astype(float)).max()))
  if start%320==0:driver.save(OUT/'progress.json',dict(completed_windows=stop,total_windows=len(raw)));print('CONNECTIVITY CACHE',stop,'/2360',flush=True)
 static.flush();dynamic.flush();assert quantization<=.00025
 driver.save(OUT/'completion.json',dict(complete=True,windows=len(raw),max_float16_absolute_error=quantization,static_sha256=driver.sha(OUT/'static_corr.npy'),dynamic_sha256=driver.sha(OUT/'local_corr.npy'),public_data_used=False,holdout_used=False));print('LOCAL CONNECTIVITY CACHE COMPLETE',flush=True)
if __name__=='__main__':main()
