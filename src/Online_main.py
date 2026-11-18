import argparse, json
from pathlib import Path
import numpy as np
from Online_utility import load_splits
from Online_features_extracting import extract
from Online_features_engineering import aggregate, causal_scores, FeaturesEngineering
from Online_anomaly_detection import AnomalyDetector
from Online_model_evaluation import calibrate, alerts, report

def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataset_name',default='SP_150MB_ratio'); p.add_argument('--data_root',default='../../LWADLS/datasets'); p.add_argument('--round',default='1'); p.add_argument('--mode',choices=['M','S'],default='M'); p.add_argument('--device',choices=['auto','cpu','cuda'],default='cpu'); p.add_argument('--seed',type=int,default=42); p.add_argument('--batch-size',type=int,default=2048); p.add_argument('--output-root',default='novaadls_online_runs'); p.add_argument('--fp-cost',type=float,default=10); p.add_argument('--fn-cost',type=float,default=20); p.add_argument('--delay-cost',type=float,default=5); a=p.parse_args()
    import torch
    device=('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else a.device
    if device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    output=Path(a.output_root)/a.dataset_name; output.mkdir(parents=True,exist_ok=True)
    print('Device:',device,flush=True)
    df=load_splits(a.data_root,a.dataset_name,a.round,a.seed)
    df=extract(df,device); seq,X,masks,scaler,events=aggregate(df); del df
    n,u,v,t=masks
    Xtr,ytr,Xte,yte,Xv,yv=FeaturesEngineering.novelty_detection_label_establishment(X,seq,X[n],X[u],seq.loc[u,'Label'].to_numpy(),n,u,v,t)
    if len(np.unique(ytr))!=2: raise ValueError('Pseudo-labels must contain both classes')
    model,standard_threshold,fit_time=AnomalyDetector.fit_model(np.asarray(Xtr),ytr,np.asarray(Xv),yv,a.mode,device)
    val_streams=[events[k] for k in seq.loc[v,'Node_block_id']]
    print('Calibrating online policy on validation only',flush=True)
    val_scores=causal_scores(model,scaler,val_streams,a.batch_size)
    policy=calibrate(val_scores,yv,standard_threshold)
    del val_scores
    print('Standard threshold:',standard_threshold,'Online policy:',policy,flush=True)
    test_streams=[events[k] for k in seq.loc[t,'Node_block_id']]; lengths=np.array([len(s) for s in test_streams])
    standard=np.where(model.predict_proba(np.asarray(Xte))[:,1]>=standard_threshold,lengths,0)
    print('Scoring causal test sequence replay',flush=True)
    online=alerts(causal_scores(model,scaler,test_streams,a.batch_size),policy['threshold'],policy['persistence'])
    report(yte,standard,online,lengths,(a.fp_cost,a.fn_cost,a.delay_cost),output)
    (output/'policies.json').write_text(json.dumps({'standard_threshold':standard_threshold,'online':policy,'fit_minutes':fit_time,'settings':vars(a)},indent=2))
    print('Results:',output/'comparison.csv')

if __name__=='__main__': main()
