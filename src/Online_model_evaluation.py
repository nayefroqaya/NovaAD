import numpy as np
import pandas as pd
from pathlib import Path

def metrics(y,steps,lengths,costs):
    y=np.asarray(y); steps=np.asarray(steps); lengths=np.asarray(lengths)
    if y.shape!=steps.shape or y.shape!=lengths.shape or (lengths<1).any(): raise ValueError('Invalid evaluation alignment')
    if ((steps<0)|(steps>lengths)).any(): raise ValueError('Invalid alert step')
    p=steps>0; tp=int((p & (y==1)).sum()); tn=int((~p & (y==0)).sum()); fp=int((p & (y==0)).sum()); fn=int((~p & (y==1)).sum())
    div=lambda a,b: a/b if b else 0.
    ratios=steps[p & (y==1)]/lengths[p & (y==1)]
    fc,nc,dc=fp*costs[0],fn*costs[1],float(ratios.sum()*costs[2])
    return {'Test sequences':len(y),'Truly anomalous sequences':tp+fn,'TP':tp,'TN':tn,'FP':fp,'FN':fn,'Precision':div(tp,tp+fp),'Recall':div(tp,tp+fn),'F1':div(2*tp,2*tp+fp+fn),'FPR':div(fp,fp+tn),'AvgRatio':float(ratios.mean()) if tp else 0., **{f'EDR@{k}':div(int((ratios<=k/100).sum()),tp+fn) for k in (25,50,75)},'FP cost':fc,'FN cost':nc,'Delay cost':dc,'Total cost':fc+nc+dc}

def alerts(scores,threshold,persistence):
    steps=[]
    for stream in scores:
        a=np.asarray(stream)>=threshold
        if len(a)<persistence: steps.append(0); continue
        hit=np.flatnonzero(np.convolve(a.astype(int),np.ones(persistence,dtype=int),mode='valid')==persistence)
        steps.append(int(hit[0])+persistence if len(hit) else 0)
    return np.asarray(steps)

def calibrate(scores,y,standard_threshold):
    # Fixed candidate grid; same F2 objective as original full-sequence threshold.
    y=np.asarray(y); lengths=np.array([len(s) for s in scores]); best=None
    for p in (1,2,3,5):
        # For a run of p consecutive scores, its minimum is the alert threshold limit.
        maxima=np.array([np.max(np.lib.stride_tricks.sliding_window_view(s,p).min(axis=1)) if len(s)>=p else -1 for s in scores])
        for t in np.unique(np.r_[np.linspace(0,1,101),standard_threshold,1.000001]):
            pred=maxima>=t; tp=int((pred & (y==1)).sum()); fp=int((pred & (y==0)).sum()); fn=int((~pred & (y==1)).sum()); f2=5*tp/(5*tp+4*fn+fp) if tp else 0.
            key=(f2,-fp,-p,float(t))
            if best is None or key>best[0]: best=(key,float(t),p)
    return {'threshold':best[1],'persistence':best[2],'validation_f2':best[0][0]}

def report(y,standard,online,lengths,costs,output):
    table=pd.DataFrame({'NovaADLS-Standard':metrics(y,standard,lengths,costs),'NovaADLS-Online':metrics(y,online,lengths,costs)})
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    table.to_csv(output/'comparison.csv',index_label='Metric'); print(table.to_string(float_format=lambda x:f'{x:.6f}'))
    return table
