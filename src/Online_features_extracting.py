"""Original feature families; learned transformations fitted on training events only."""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from scipy.sparse import diags

def extract(df,device='cpu'):
    from sentence_transformers import SentenceTransformer
    from transformers import pipeline
    df=df.copy(); texts=df.processed_EventTemplate.astype(str)
    train=df.Temp_label.isin([0,999]).to_numpy()
    unique, inverse=np.unique(texts.to_numpy(dtype=str),return_inverse=True)
    counts=np.bincount(inverse[train],minlength=len(unique)); known=counts>0
    print('Extracting sentiment and semantics',flush=True)
    sentiment=pipeline('sentiment-analysis',model='siebert/sentiment-roberta-large-english',device=0 if device=='cuda' else -1)
    sent=[]
    for i in range(0,len(unique),64):
        sent.extend(r['label'] for r in sentiment(unique[i:i+64].tolist(),batch_size=32,truncation=True))
    df['sentiment_label']=np.asarray(sent)[inverse]
    encoder=SentenceTransformer('bert-base-nli-mean-tokens',device=device)
    emb=encoder.encode(unique.tolist(),batch_size=64,show_progress_bar=True,convert_to_numpy=True)
    # Weighted moments are mathematically equivalent to expanding each training
    # embedding by its event count, without an N_events x 768 allocation.
    train_emb=emb[known].astype(np.float64); weights=counts[known].astype(np.float64)
    sem_scaler=StandardScaler().fit(train_emb,sample_weight=weights)
    z=sem_scaler.transform(train_emb); center=np.average(z,axis=0,weights=weights)
    z-=center
    covariance=(z.T @ (z*weights[:,None]))/max(1,weights.sum()-1)
    eigenvalues,eigenvectors=np.linalg.eigh(covariance)
    max_k=min(40,int(weights.sum()),z.shape[1]); candidates=[k for k in [5,10,15,20,25,30,35,40] if k<=max_k] or [max_k]
    # Largest candidate maximizes cumulative explained variance (original rule).
    k=candidates[-1]; components=eigenvectors[:,-k:][:,::-1].T
    # Deterministic sign convention for eigenvectors.
    signs=np.sign(components[np.arange(k),np.abs(components).argmax(axis=1)]); signs[signs==0]=1
    components*=signs[:,None]
    reduced=((sem_scaler.transform(emb)-center) @ components.T).astype(np.float32)
    del train_emb,z,covariance
    df['reduced_embedding']=list(reduced[inverse])
    print('Fitting TF-IDF and LDA on training templates only',flush=True)
    tf=TfidfVectorizer(stop_words='english').fit(unique[known].tolist())
    weighted=diags(counts[known]) @ tf.transform(unique[known].tolist())
    lda=LatentDirichletAllocation(learning_method='online',max_iter=100,random_state=0)
    cv=min(5,int(known.sum()))
    if cv<2: raise ValueError('Insufficient unique training templates for LDA')
    search=GridSearchCV(lda,{'n_components':[2,3,5,7,9,13,15]},cv=cv,error_score='raise',verbose=2).fit(weighted)
    lda=search.best_estimator_
    # Use frozen TRAIN frequencies for known texts; unseen templates weight 1.
    topics=lda.transform(diags(np.maximum(counts,1)) @ tf.transform(unique.tolist()))
    df['Dominant_Topic']=topics.argmax(axis=1)[inverse]
    logs=np.zeros_like(topics); np.log2(topics,out=logs,where=topics>0)
    df['entropy']=(-(topics*logs).sum(axis=1))[inverse]
    df['num_words']=texts.str.split().str.len(); df['Character_Count']=texts.str.len()
    for c in ['year','month','day','hour','minute','second']: df[c]=getattr(df.Timestamp.dt,c)
    return df
