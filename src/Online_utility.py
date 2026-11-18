from pathlib import Path
import numpy as np
import pandas as pd

def load_splits(root,dataset,round_id,seed):
    base=Path(root)/dataset/f'{round_id}_{dataset}_Splitted_Datasets'
    frames=[]; seen=set(); rng=np.random.default_rng(seed)
    for name,tag in [('train',999),('val',777),('test',888)]:
        df=pd.read_pickle(base/f'{name}_df.pkl').copy()
        req=['Node_block_id','Label','Timestamp','processed_EventTemplate']
        if not set(req)<=set(df): raise ValueError(f'{name}: missing {set(req)-set(df)}')
        if df[req].isna().any().any(): raise ValueError(f'{name}: null required values')
        df['Node_block_id']=df.Node_block_id.astype(str)
        ids=set(df.Node_block_id)
        if seen & ids: raise ValueError(f'{name}: overlapping block IDs')
        seen.update(ids)
        def norm(x):
            s=str(x).strip().lower()
            if s in ('normal','-','0','0.0','false'): return 'Normal'
            if s in ('anomaly','abnormal','anomalous','1','1.0','true'): return 'Anomaly'
            raise ValueError(f'Unexpected Label {x!r}')
        df['Label']=df.Label.map(norm)
        df['Timestamp']=pd.to_datetime(df.Timestamp,errors='raise')
        df=df.sort_values(['Node_block_id','Timestamp'],kind='stable').reset_index(drop=True)
        df['Temp_label']=tag
        if name=='train':
            normal=df.groupby('Node_block_id').Label.apply(lambda x:(x=='Normal').all())
            candidates=normal.index[normal & ~normal.index.str.contains('UNKNOWN',case=False)].to_numpy()
            if len(candidates)<2: raise ValueError('Not enough labeled-normal candidates')
            selected=rng.choice(candidates,len(candidates)//2,replace=False)
            df.loc[df.Node_block_id.isin(selected),'Temp_label']=0
        print(f'{name}: {len(ids)} sequences, {len(df)} events',flush=True)
        frames.append(df)
    return pd.concat(frames,ignore_index=True)
