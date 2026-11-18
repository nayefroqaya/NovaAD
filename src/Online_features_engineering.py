import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report

FEATURES = ['sentiment_label','Dominant_Topic','num_words','Character_Count','entropy','year','month','day','hour','minute','second']

def aggregate(df):
    # The uploaded version uses SUM, not mean.
    df=df.copy()
    enc=LabelEncoder().fit(df.loc[df.Temp_label.isin([0,999]),'sentiment_label'])
    mapping={v:i for i,v in enumerate(enc.classes_)}
    df['sentiment_label']=df.sentiment_label.map(mapping).fillna(-1)
    rows=[]; events={}
    for key,g in df.groupby('Node_block_id',sort=True,observed=True):
        if g.Temp_label.nunique()!=1: raise ValueError(f'Mixed split/portion: {key}')
        a=np.column_stack((np.stack(g.reduced_embedding),g[FEATURES].to_numpy(dtype=np.float32))).astype(np.float32)
        if not np.isfinite(a).all(): raise ValueError(f'Nonfinite features: {key}')
        tag=int(g.Temp_label.iloc[0]); y=int((g.Label!='Normal').any())
        rows.append((key,tag,y,a.sum(axis=0)))
        if tag in (777,888): events[key]=a
    seq=pd.DataFrame(rows,columns=['Node_block_id','Temp_label','Label','features'])
    # Match original order: labeled train, unlabeled train, test, validation.
    seq=pd.concat([seq[seq.Temp_label==t] for t in (0,999,888,777)],ignore_index=True)
    masks=tuple(seq.Temp_label.eq(t).to_numpy() for t in (0,999,777,888))
    if not all(m.any() for m in masks): raise ValueError('Empty data portion')
    raw=np.stack(seq.features); scaler=StandardScaler().fit(raw[masks[0]|masks[1]])
    return seq,scaler.transform(raw),masks,scaler,events

def causal_scores(model,scaler,streams,batch_size=2048):
    # One incremental sum per sequence; no reprocessing of earlier events.
    # Batching scores is an offline replay of causal inference, not live latency.
    out=[]; pending=[]; locations=[]
    def flush():
        if not pending: return
        probs=model.predict_proba(scaler.transform(np.stack(pending)))[:,1]
        for (i,j),v in zip(locations,probs): out[i][j]=v
        pending.clear(); locations.clear()
    for i,a in enumerate(streams):
        out.append(np.empty(len(a),dtype=float)); state=np.zeros(a.shape[1],dtype=np.float32)
        for j,event in enumerate(a):
            state+=event; pending.append(state.copy()); locations.append((i,j))
            if len(pending)>=batch_size: flush()
    flush(); return out

class FeaturesEngineering:
    @staticmethod
    def novelty_detection_label_establishment(X_sequences_df, sequences_df, X_train_normal_labelled, X_unlabeled_train,
                                              ground_truth_unlabeled_data_from_train, m_train_normal, m_train_unlabeled,
                                              m_val, m_test):

        # Parameter grid for One-Class SVM
        gamma_values = np.linspace(0.2, 0.5, 5)
        nu_values = np.linspace(0.01, 0.08, 5)

        # Candidate alpha values for hybrid score
        alpha_values = np.linspace(0.5, 1, 5)  # 0.0, 0.1, ..., 1.0

        best_params = None
        best_alpha = None
        best_score = -np.inf  # Higher hybrid score is better

        '''
        # Grid Search Over Gamma, Nu, and Alpha
        for gamma, nu in product(gamma_values, nu_values):
            print(f'Testing parameters: Gamma={gamma}, Nu={nu}')

            # Train One-Class SVM on normal labeled data
            oc_svm = OneClassSVM(kernel="rbf", gamma=gamma, nu=nu)
            oc_svm.fit(X_train_normal_labelled)

            # Predict labels on unlabeled data (1 = normal, -1 = anomaly)
            preds = oc_svm.predict(X_unlabeled_train)

            # Compute LOF scores for predicted normal points
            if np.sum(preds == 1) > 1:  # At least 2 points to compute LOF
                lof = LocalOutlierFactor(n_neighbors=20)
                lof_scores = -lof.fit_predict(X_unlabeled_train[preds == 1])
                lof_mean = np.mean(lof_scores)
            else:
                lof_mean = 0  # fallback if no normal points

            # Compute Silhouette score
            if len(set(preds)) > 1:
                silhouette = silhouette_score(X_unlabeled_train, preds)
            else:
                silhouette = -1  # fallback if only one cluster

            # Sweep over alpha to compute hybrid score
            for alpha in alpha_values:
                hybrid_score = alpha * lof_mean + (1 - alpha) * silhouette

                if hybrid_score > best_score:
                    best_score = hybrid_score
                    best_params = (gamma, nu)
                    best_alpha = alpha

        print(f"Optimal parameters found: Gamma={best_params[0]}, Nu={best_params[1]}, Alpha={best_alpha}")
        '''

        # Train final model with optimal Gamma & Nu
        #oc_svm_final = OneClassSVM(kernel='rbf', gamma=best_params[0], nu=best_params[1])
        oc_svm_final = OneClassSVM(kernel='rbf', gamma= 0.01, nu=0.45)

        oc_svm_final.fit(X_train_normal_labelled)
        # Make predictions
        y_pred = oc_svm_final.predict(X_unlabeled_train)
        pseudo_labels = np.where(y_pred == -1, 1, 0)  # Map -1 (outliers) to 1 (anomaly), 1 (inliers) to 0 (normal)

        # Ensure matching lengths
        if len(ground_truth_unlabeled_data_from_train) != len(pseudo_labels):
            print("Error: Length mismatch between ground truth and predictions")
            return None

        # Print classification report for pseudo-labels
        print("\nPseudo-Labeling Classification Report (One-Class SVM):")
        print(classification_report(ground_truth_unlabeled_data_from_train, pseudo_labels,
                                    target_names=["Normal", "Anomaly"]))

        # Prepare the Training Data ----------------------------------------------------------------------------
        # ---- build train normal df (keep ground truth + final label) ----
        df_train_normal = sequences_df[sequences_df['Temp_label'] == 0][['Temp_label', 'Label']].copy()
        df_train_normal['Final_Label'] = df_train_normal['Label'].values
        df_train_normal['scaled_features'] = list(X_sequences_df[m_train_normal])

        # ---- build train unlabeled df (pseudo labels) ----
        df_train_unlabeled = sequences_df[sequences_df['Temp_label'] == 999][['Temp_label', 'Label']].copy()
        df_train_unlabeled['Final_Label'] = pseudo_labels
        df_train_unlabeled['scaled_features'] = list(X_sequences_df[m_train_unlabeled])

        # ---- combine ----
        df_final = pd.concat([df_train_normal, df_train_unlabeled], ignore_index=True)

        # ---- evaluation on full train ----
        y_true = df_final['Label'].values
        y_pred = df_final['Final_Label'].values
        print("\nFull Training Data Classification Report (One-Class SVM):")
        print(classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"]))

        # ---- test df ----
        df_test = sequences_df[sequences_df['Temp_label'] == 888][['Temp_label', 'Label']].copy()
        df_test['scaled_features'] = list(X_sequences_df[m_test])

        # ---- val df ----
        df_val = sequences_df[sequences_df['Temp_label'] == 777][['Temp_label', 'Label']].copy()
        df_val['scaled_features'] = list(X_sequences_df[m_val])

        # ---- check test label distribution ----
        xx_0 = df_test[df_test['Label'] == 0]
        xx_1 = df_test[df_test['Label'] == 1]
        if len(xx_0) == 0 or len(xx_1) == 0:
            print('Error: Test data label distribution issue')
            exit()

        # ---- extract features and labels (SCALED) ----
        X_train = df_final['scaled_features'].tolist()
        y_train = df_final['Final_Label'].values

        X_test = df_test['scaled_features'].tolist()
        y_test_truth = df_test['Label'].values

        X_val = df_val['scaled_features'].tolist()
        y_val_truth = df_val['Label'].values

        return X_train, y_train, X_test, y_test_truth, X_val, y_val_truth

