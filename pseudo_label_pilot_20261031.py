"""Frozen-teacher pseudo labels and development-only transductive pilot.
Public signals are not used for fitting. No true public labels are available.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import csv, hashlib, json
from pathlib import Path
import numpy as np
from scipy.special import expit, logit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from openpyxl import Workbook
import driver

OUT=driver.JOB/'pseudo_label_pilot_20261031'
OUT.mkdir(exist_ok=True)
RELEASE=Path('/workspace/rank16_release/EEG_RANK16_20261030')

def main():
    spec=dict(confidence_threshold=.85,pseudo_weight=.2,C=.01,teacher_weight=.9,
              seed=1031,public_training=False,official_submission=False,
              rule_permission_for_public_training='not_yet_confirmed',
              evaluation='exploratory reused development folds; outer EEG used without labels for adaptation',
              teacher='frozen rank16',script_sha256=driver.sha(__file__))
    driver.save(OUT/'protocol.json',spec)
    # Generate annotations from existing frozen predictions, never true labels.
    pfile=RELEASE/'predictions/release_prediction/fold_predictions.npz'
    with np.load(pfile) as a:
        q=a['probability']; people=a['subject']; trials=a['trial']; fq=a['fold_probability']
    assert len(q)==80 and np.isfinite(q).all() and ((q>=0)&(q<=1)).all()
    columns=['user_id','trial_id','predicted_probability','pseudo_label_NOT_ground_truth','confidence','eligible_at_0.85','fold_vote_agreement']
    records=[]
    for i in range(80):
        hard=int(q[i]>=.5); confidence=float(max(q[i],1-q[i]))
        records.append([str(people[i]),int(trials[i]),float(q[i]),hard,confidence,confidence>=.85,float(np.mean((fq[:,i]>=.5)==hard))])
    with (OUT/'public80_PSEUDOLABELS_NOT_TRUTH.csv').open('w',encoding='utf-8-sig',newline='') as h:
        w=csv.writer(h);w.writerow(columns);w.writerows(records)
    book=Workbook();sheet=book.active;sheet.title='PSEUDO_NOT_GROUND_TRUTH';sheet.append(columns)
    for row in records:sheet.append(row)
    book.save(OUT/'public80_PSEUDOLABELS_NOT_TRUTH.xlsx')
    # Source-only replay: use fixed DE summaries as a lightweight student.
    with np.load(driver.ART/'features.npz') as a:
        subject=a['subject']; powers=a['powers']; sequence=a['sequence']
    with np.load(driver.CONTROL/'full_evaluation_rows.npz') as a:
        mapping=a['model_row_to_full_row']; full_subject=a['subject']; full_y=a['label']; diagnosis=a['diagnosis']
    with np.load(driver.JOB/'pooled_rank_bag_20261030/subbag16_rawmean_calibrated_oof.npz') as a: teacher=a['probability'].copy()
    de=sequence.mean(axis=1).reshape(len(subject),-1).astype(float)
    centered=de.copy()
    for s in np.unique(subject):
        ix=np.flatnonzero(subject==s);centered[ix]-=de[ix].mean(axis=0)
    x=np.column_stack([de,centered]); baseline=teacher.copy(); adapted=teacher.copy(); details=[]
    for fold in driver.protocol()['folds']:
        tr=np.flatnonzero(np.isin(subject,fold['training_subjects'])); va=np.flatnonzero(np.isin(subject,fold['validation_subjects']))
        assert not set(subject[tr]) & set(subject[va])
        # Only source labels are available until all fits have completed.
        ytr=full_y[mapping[tr]].copy(); tq=teacher[mapping[va]]; hard=(tq>=.5).astype(int)
        selected=np.maximum(tq,1-tq)>=spec['confidence_threshold']
        scaler=StandardScaler().fit(x[tr]); xt=scaler.transform(x[tr]); xv=scaler.transform(x[va])
        model=LogisticRegression(C=spec['C'],max_iter=1000,solver='lbfgs',random_state=1031).fit(xt,ytr)
        pb=model.predict_proba(xv)[:,1]
        trainx=np.vstack([xt,xv[selected]]); trainy=np.r_[ytr,hard[selected]]
        weights=np.r_[np.ones(len(tr)),np.full(selected.sum(),spec['pseudo_weight'])]
        model2=LogisticRegression(C=spec['C'],max_iter=1000,solver='lbfgs',random_state=1031).fit(trainx,trainy,sample_weight=weights)
        pa=model2.predict_proba(xv)[:,1]
        def blend(z):return expit(.9*logit(np.clip(tq,1e-6,1-1e-6))+.1*logit(np.clip(z,1e-6,1-1e-6)))
        baseline[mapping[va]]=blend(pb); adapted[mapping[va]]=blend(pa)
        details.append(dict(fold=fold['fold'],pseudo_rows=int(selected.sum()),source_rows=len(tr),query_rows=len(va),pseudo_positives=int(hard[selected].sum())))
        print('PILOT FOLD',fold['fold'],'/5 DONE; pseudo rows',selected.sum(),flush=True)
    # First access to query truth for scoring: no optimization follows this.
    dep=diagnosis==1; results={}
    for name,pred in [('frozen_teacher',teacher),('teacher_plus_source_student',baseline),('teacher_plus_pseudo_student',adapted)]:
        results[name]=dict(pooled_auc=float(roc_auc_score(full_y,pred)),equal_clinical_mass_pooled_auc=float(roc_auc_score(full_y,pred,sample_weight=np.where(dep,2.,1.))),DEP_auc=float(roc_auc_score(full_y[dep],pred[dep])))
    np.savez_compressed(OUT/'development_predictions.npz',teacher=teacher,source_student_blend=baseline,pseudo_student_blend=adapted)
    driver.save(OUT/'summary.json',dict(results=results,folds=details,public_pseudolabel_count=80,public_eligible_count=sum(r[5] for r in records),true_public_labels_available=False,public_prediction_sha256=hashlib.sha256(pfile.read_bytes()).hexdigest(),public_model_not_adapted=True,baseline_not_replaced=True,independent_confirmation=False,official_feedback=88.13))
    print(json.dumps(results),flush=True)
if __name__=='__main__':main()
