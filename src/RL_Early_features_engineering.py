# RL_Early: NovaADLS Early baseline. Minimal prefix-aware extension of the original implementation.

import warnings
from itertools import product
import colorama
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, classification_report
from sklearn.metrics import silhouette_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

warnings.filterwarnings('ignore')
colorama.init()
GREEN = colorama.Fore.GREEN
GRAY = colorama.Fore.LIGHTBLACK_EX
RESET = colorama.Fore.RESET
YELLOW = colorama.Fore.YELLOW


class FeaturesEngineering:

    @staticmethod
    def summing_Train_test_val_together(dataset, df_train_with_test_with_val):
        df_train_with_test_with_val = df_train_with_test_with_val.dropna(how='all')
        df_train_with_test_with_val = df_train_with_test_with_val.reset_index(drop=True)

        # Define safe aggregation function
        def safe_stack(feature_list):
            arrays = [np.asarray(f, dtype=np.float32) for f in feature_list if f is not None and len(f) > 0]
            if len(arrays) == 0:
                first = feature_list.iloc[0] if len(feature_list) > 0 else np.zeros(51, dtype=np.float32)
                return np.zeros(len(first), dtype=np.float32) if first is not None else np.zeros(51, dtype=np.float32)

                # Return zeros with the same length as any feature vector
            #              return np.zeros(len(feature_list[0]), dtype=np.float32)
            # return np.mean(np.stack(arrays), axis=0)
            # return np.max(np.stack(arrays), axis=0)
            return np.sum(np.stack(arrays), axis=0)

        log_normal_labelled = df_train_with_test_with_val[df_train_with_test_with_val['Temp_label'] == 0].copy()
        # 2️⃣ Remove UNKNOWN blocks (case-insensitive)
        log_normal_labelled = log_normal_labelled[
            ~log_normal_labelled['Node_block_id'].astype(str).str.contains("UNKNOWN", case=False,
                                                                           na=False)].reset_index(drop=True)

        log_remain_normal_anomaly_unlabelled = df_train_with_test_with_val[
            (df_train_with_test_with_val['Temp_label'] == 999)].copy()  # Normal/Anomaly logs - unlabeled train data
        log_test_unlabelled = df_train_with_test_with_val[
            (df_train_with_test_with_val['Temp_label'] == 888)].copy()  # Normal/Anomaly logs - unlabeled test data
        log_val_labeled = df_train_with_test_with_val[
            (df_train_with_test_with_val['Temp_label'] == 777)].copy()  # Normal/Anomaly logs - unlabeled test data

        # (1) Train Normal logs labelled -----------------------------------------------------------------------
        # 4️⃣ Aggregate features by Node_block_id
        summed_df_normal_labelled_train = log_normal_labelled.groupby('Node_block_id')['features'].apply(
            safe_stack).reset_index()

        # 5️⃣ Assign sequence labels (normal/anomaly) safely
        sequence_labels_normal_labelled = log_normal_labelled.groupby('Node_block_id')['Label'].apply(
            lambda x: 'anomaly' if any(lbl != 'Normal' for lbl in x) else 'normal').reset_index()

        # 6️⃣ Merge features with labeel
        summed_df_normal_labelled_train = summed_df_normal_labelled_train.merge(sequence_labels_normal_labelled,
                                                                                on='Node_block_id')
        summed_df_normal_labelled_train['Temp_label'] = 0  # Normal

        # Validation checks
        xx_anomaly = summed_df_normal_labelled_train[(summed_df_normal_labelled_train['Label'] == 'anomaly')]
        yy_normal = summed_df_normal_labelled_train[(summed_df_normal_labelled_train['Label'] == 'normal')]

        if len(xx_anomaly) != 0 or len(yy_normal) == 0:
            print('Error in summing_Train_test_val_together function: Normal data validation failed')
            exit()

        if len(yy_normal) != len(summed_df_normal_labelled_train):
            print('Error: Length mismatch in normal labelled data')
            exit()
        # (2) Train - Normal and Abnormal logs unlabelled ------------------------------------------------------
        summed_df_combine_unlabelled_train = log_remain_normal_anomaly_unlabelled.groupby('Node_block_id')[
            'features'].apply(safe_stack).reset_index()

        # 5️⃣ Assign sequence labels (normal/anomaly) safely
        sequence_labels_combine_unlabelled = log_remain_normal_anomaly_unlabelled.groupby('Node_block_id')[
            'Label'].apply(lambda x: 'anomaly' if any(lbl != 'Normal' for lbl in x) else 'normal').reset_index()

        # 6️⃣ Merge features with labeel
        # Merging the summed feature vectors with their respective labels
        summed_df_combine_unlabelled_train = summed_df_combine_unlabelled_train.merge(
            sequence_labels_combine_unlabelled, on='Node_block_id')
        summed_df_combine_unlabelled_train['Temp_label'] = 999  # Normal and anomaly unlabeled

        # Validation checks
        xx_anomaly = summed_df_combine_unlabelled_train[(summed_df_combine_unlabelled_train['Label'] == 'anomaly')]
        yy_normal = summed_df_combine_unlabelled_train[(summed_df_combine_unlabelled_train['Label'] == 'normal')]

        if len(xx_anomaly) == 0 or len(yy_normal) == 0:
            print('Error in summing_Train_test_together function: Unlabeled data validation failed')
            print(len(yy_normal))
            print(len(xx_anomaly))
            ##exit()

        summed_df_train = pd.concat([summed_df_normal_labelled_train, summed_df_combine_unlabelled_train])

        # (3) Test dataset: ------------------------------------------------------------------------------------
        summed_df_combine_unlabelled_test = log_test_unlabelled.groupby('Node_block_id')['features'].apply(
            safe_stack).reset_index()
        sequence_labels_combine_unlabelled_test = log_test_unlabelled.groupby('Node_block_id')['Label'].apply(
            lambda x: 'anomaly' if any(lbl != 'Normal' for lbl in x) else 'normal').reset_index()
        # Merging the summed feature vectors with their respective labels
        summed_df_combine_unlabelled_test = summed_df_combine_unlabelled_test.merge(
            sequence_labels_combine_unlabelled_test, on='Node_block_id')
        summed_df_combine_unlabelled_test['Temp_label'] = 888  # Test normal and anomaly unlabeled
        summed_df_test = summed_df_combine_unlabelled_test

        # (4) Val dataset: ------------------------------------------------------------------------------------
        summed_df_val_dataset = log_val_labeled.groupby('Node_block_id')['features'].apply(safe_stack).reset_index()
        sequence_labels_val = log_val_labeled.groupby('Node_block_id')['Label'].apply(
            lambda x: 'anomaly' if any(lbl != 'Normal' for lbl in x) else 'normal').reset_index()
        # Merging the summed feature vectors with their respective labels
        summed_df_val = summed_df_val_dataset.merge(sequence_labels_val, on='Node_block_id')
        summed_df_val['Temp_label'] = 777  # Test normal and anomaly unlabeled

        # Final processing and combination
        summed_df_train = summed_df_train.reset_index(drop=True)
        summed_df_test = summed_df_test.reset_index(drop=True)
        summed_df_val = summed_df_val.reset_index(drop=True)

        summ_train_test_val_combine = pd.concat([summed_df_train, summed_df_test, summed_df_val])
        summ_train_test_val_combine = summ_train_test_val_combine.reset_index(drop=True)

        print('Feature aggregation completed successfully')
        return summ_train_test_val_combine

    @staticmethod
    def preparing_features_training_combined_labeled_unlabeled(dataset, df_features_all_train_with_test_final):
        df_features_all_train_with_test_final['features'] = df_features_all_train_with_test_final['reduced_embedding']

        # Append additional features to the feature vector
        feature_columns = ['sentiment_label', 'Dominant_Topic', 'num_words', 'Character_Count', 'entropy', 'year',
                           'month', 'day', 'hour', 'minute', 'second']

        for col in feature_columns:
            df_features_all_train_with_test_final['features'] = df_features_all_train_with_test_final.apply(
                lambda row: row['features'] + [row[col]], axis=1)

        return df_features_all_train_with_test_final

    @staticmethod
    def features_aggregation_transformation(final_train_with_test_with_val, dataset):
        # Function to handle replacement of None, NaN, or other values
        def replace_none_or_nan(x):
            if x is None:
                return np.nan
            if isinstance(x, list):
                return [np.nan if elem in ['', 'None', 'NaN'] else elem for elem in x]
            if isinstance(x, (str, int, float)) and x in ['', 'None', 'NaN']:
                return np.nan
            return x

        final_train_with_test_with_val = final_train_with_test_with_val.applymap(replace_none_or_nan)
        final_train_with_test_with_val = final_train_with_test_with_val.dropna(how='all')
        final_train_with_test_with_val = final_train_with_test_with_val.reset_index(drop=True)

        # Optimize data types
        final_train_with_test_with_val = final_train_with_test_with_val.astype({
            col: 'float32' if final_train_with_test_with_val[col].dtype == 'float64' else 'int32' if
            final_train_with_test_with_val[col].dtype == 'int64' else final_train_with_test_with_val[col].dtype for col
            in final_train_with_test_with_val.columns})

        # Encode sentiment labels
        label_encoder_sentiment = LabelEncoder()
        final_train_with_test_with_val['sentiment_label'] = label_encoder_sentiment.fit_transform(
            final_train_with_test_with_val['sentiment_label'])

        # Convert embeddings to lists
        final_train_with_test_with_val['reduced_embedding'] = final_train_with_test_with_val['reduced_embedding'].apply(
            lambda x: x.tolist())

        # Prepare features and aggregate
        df_train_with_test_with_val = FeaturesEngineering.preparing_features_training_combined_labeled_unlabeled(
            dataset, final_train_with_test_with_val)
        sequences_df = FeaturesEngineering.summing_Train_test_val_together(dataset, df_train_with_test_with_val)

        # -------new
        # Convert features into numpy array (keeps same row order as sequences_df)
        Train_Test_val_df_lst = np.vstack(sequences_df['features'].values)

        # Convert ground-truth label to binary
        sequences_df['Label'] = sequences_df['Label'].apply(lambda x: 0 if x == 'normal' else 1)

        # Validation check
        xx_0 = sequences_df[sequences_df['Label'] == 0]
        xx_1 = sequences_df[sequences_df['Label'] == 1]
        if len(xx_0) == 0 or len(xx_1) == 0:
            print('Error: Truth label distribution issue')
            exit()

        # -------------------------
        # Create masks
        # -------------------------
        m_train_normal = (sequences_df['Temp_label'] == 0).to_numpy()
        m_train_unlabeled = (sequences_df['Temp_label'] == 999).to_numpy()
        m_val = (sequences_df['Temp_label'] == 777).to_numpy()
        m_test = (sequences_df['Temp_label'] == 888).to_numpy()

        # Full train mask (for scaler fitting)
        m_train = m_train_normal | m_train_unlabeled

        # -------------------------
        # Scale features
        # -------------------------
        scaler_data = StandardScaler()

        # Fit ONLY on training data
        scaled_train = scaler_data.fit_transform(Train_Test_val_df_lst[m_train])

        # Transform val and test using train statistics
        scaled_val = scaler_data.transform(Train_Test_val_df_lst[m_val])
        scaled_test = scaler_data.transform(Train_Test_val_df_lst[m_test])

        print('Feature scaling completed')

        # Rebuild full scaled dataset (same row order)
        scaled_Train_Test_val_df = np.empty_like(Train_Test_val_df_lst, dtype=float)

        scaled_Train_Test_val_df[m_train] = scaled_train
        scaled_Train_Test_val_df[m_val] = scaled_val
        scaled_Train_Test_val_df[m_test] = scaled_test

        # Final X and y
        X_sequences_df = scaled_Train_Test_val_df
        y_sequences_df = sequences_df['Temp_label']

        # -------------------------
        # Return everything
        # -------------------------
        return sequences_df, X_sequences_df, y_sequences_df, m_train_normal, m_train_unlabeled, m_val, m_test

        # return sequences_df, X_sequences_df, y_sequences_df

        '''
        # Apply Standard Scaler to the summed feature vectors
        Train_Test_val_df_lst = np.vstack(sequences_df['features'].values)
        sequences_df['Label'] = sequences_df['Label'].apply(lambda x: 0 if x == 'normal' else 1)

        # Validation checks
        xx_0 = sequences_df[(sequences_df['Label'] == 0)]
        xx_1 = sequences_df[(sequences_df['Label'] == 1)]
        if len(xx_0) == 0 or len(xx_1) == 0:
            print('Error: Truth label distribution issue')
            exit()

        scaler_data = StandardScaler()
        scaled_Train_Test_val_df = scaler_data.fit_transform(Train_Test_val_df_lst)
        print('Feature scaling completed')

        # Convert features into a NumPy array
        X_sequences_df = scaled_Train_Test_val_df
        y_sequences_df = sequences_df['Temp_label']

        return sequences_df, X_sequences_df, y_sequences_df
        '''


    @staticmethod
    def features_aggregation_transformation_early(
            final_train_with_test_with_val,
            dataset,
            train_prefix_fractions=(0.25, 0.50, 0.75),
            test_stride=1):
        """
        Minimal-change NovaADLS-Early aggregation.

        TRAIN:
          - retain every original full sequence vector
          - add 25%, 50%, and 75% prefix vectors
          - preserve Temp_label (0 normal-labelled / 999 unlabeled)

        DEV:
          - unchanged full-sequence vectors only

        TEST:
          - cumulative prefix vectors at every `test_stride` event
          - the final prefix is always included
          - metadata is kept so main.py can recover first-alert timing

        The underlying event-level feature representation and sum aggregation
        are unchanged.
        """

        def replace_none_or_nan(x):
            if x is None:
                return np.nan
            if isinstance(x, list):
                return [np.nan if elem in ['', 'None', 'NaN'] else elem for elem in x]
            if isinstance(x, (str, int, float)) and x in ['', 'None', 'NaN']:
                return np.nan
            return x

        def event_is_anomaly(value):
            if isinstance(value, (bool, np.bool_)):
                return bool(value)
            if isinstance(value, (int, float, np.integer, np.floating)):
                return float(value) != 0.0
            s = str(value).strip().lower()
            return s not in {"normal", "-", "0", "0.0", "false", "benign", "n"}

        df = final_train_with_test_with_val.copy()
        df = df.applymap(replace_none_or_nan)
        df = df.dropna(how='all').reset_index(drop=True)

        # Keep original feature preparation unchanged.
        label_encoder_sentiment = LabelEncoder()
        df['sentiment_label'] = label_encoder_sentiment.fit_transform(df['sentiment_label'])

        df['reduced_embedding'] = df['reduced_embedding'].apply(
            lambda x: x.tolist() if hasattr(x, 'tolist') else list(x)
        )

        df = FeaturesEngineering.preparing_features_training_combined_labeled_unlabeled(
            dataset, df
        )

        if 'Timestamp' in df.columns:
            df = df.sort_values(['Node_block_id', 'Timestamp']).reset_index(drop=True)

        rows = []

        def add_row(original_id, temp_label, vector, label, end_pos, seq_len,
                    suffix, is_full, is_train_prefix):
            rows.append({
                'Node_block_id': f"{original_id}{suffix}",
                'Original_Node_block_id': original_id,
                'features': np.asarray(vector, dtype=np.float32),
                'Label': int(label),
                'Temp_label': int(temp_label),
                'Prefix_End': int(end_pos),
                'Sequence_Length': int(seq_len),
                'Is_Full': bool(is_full),
                'Is_Train_Prefix': bool(is_train_prefix),
            })

        stride = max(1, int(test_stride))

        for original_id, group in df.groupby('Node_block_id', sort=False):
            group = group.reset_index(drop=True)
            temp_label = int(group['Temp_label'].iloc[0])
            feature_matrix = np.stack(
                [np.asarray(v, dtype=np.float32) for v in group['features'].values]
            )
            cumulative = np.cumsum(feature_matrix, axis=0)
            seq_len = len(group)

            event_labels = [event_is_anomaly(v) for v in group['Label'].values]
            full_label = 1 if any(event_labels) else 0

            # TRAIN: original full vector + 25/50/75% prefix vectors
            if temp_label in (0, 999):
                add_row(
                    original_id, temp_label, cumulative[-1], full_label,
                    seq_len, seq_len, "__FULL", True, False
                )

                used = set()
                for frac in train_prefix_fractions:
                    end_pos = int(np.ceil(float(frac) * seq_len))
                    end_pos = max(1, min(seq_len, end_pos))
                    if end_pos >= seq_len or end_pos in used:
                        continue
                    used.add(end_pos)

                    # Ground truth here is used only for diagnostics/pseudo-label
                    # quality. The final training label for Temp_label=999 is
                    # still produced by the existing One-Class SVM.
                    observed_label = 1 if any(event_labels[:end_pos]) else 0
                    add_row(
                        original_id, temp_label, cumulative[end_pos - 1],
                        observed_label, end_pos, seq_len,
                        f"__TRP{end_pos}", False, True
                    )

            # DEV: unchanged full sequence only
            elif temp_label == 777:
                add_row(
                    original_id, temp_label, cumulative[-1], full_label,
                    seq_len, seq_len, "__VAL_FULL", True, False
                )

            # TEST: sequential cumulative prefix vectors
            elif temp_label == 888:
                endpoints = list(range(1, seq_len + 1, stride))
                if endpoints[-1] != seq_len:
                    endpoints.append(seq_len)

                for end_pos in endpoints:
                    add_row(
                        original_id, temp_label, cumulative[end_pos - 1],
                        full_label, end_pos, seq_len,
                        f"__TEP{end_pos}", end_pos == seq_len, False
                    )

        sequences_df = pd.DataFrame(rows)

        # Validation
        if sequences_df.empty:
            raise ValueError("Early aggregation produced no sequence vectors.")

        X_all = np.vstack(sequences_df['features'].values)

        m_train_normal = (sequences_df['Temp_label'] == 0).to_numpy()
        m_train_unlabeled = (sequences_df['Temp_label'] == 999).to_numpy()
        m_val = (sequences_df['Temp_label'] == 777).to_numpy()
        m_test = (sequences_df['Temp_label'] == 888).to_numpy()
        m_train = m_train_normal | m_train_unlabeled

        scaler_data = StandardScaler()
        scaled_train = scaler_data.fit_transform(X_all[m_train])
        scaled_val = scaler_data.transform(X_all[m_val])
        scaled_test = scaler_data.transform(X_all[m_test])

        X_sequences_df = np.empty_like(X_all, dtype=float)
        X_sequences_df[m_train] = scaled_train
        X_sequences_df[m_val] = scaled_val
        X_sequences_df[m_test] = scaled_test

        y_sequences_df = sequences_df['Temp_label']

        print("\\n" + "=" * 72)
        print("NOVAADLS-EARLY PREFIX-AWARE AGGREGATION")
        print("=" * 72)
        print("TRAIN full vectors      :", int(np.sum(m_train & sequences_df['Is_Full'].to_numpy())))
        print("TRAIN prefix vectors    :", int(np.sum(m_train & sequences_df['Is_Train_Prefix'].to_numpy())))
        print("DEV full vectors        :", int(np.sum(m_val)))
        print("TEST prefix vectors     :", int(np.sum(m_test)))
        print("Train prefix fractions  :", list(train_prefix_fractions))
        print("Test prefix stride      :", stride)
        print("=" * 72)

        return (
            sequences_df, X_sequences_df, y_sequences_df,
            m_train_normal, m_train_unlabeled, m_val, m_test
        )


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
        oc_svm_final = OneClassSVM(kernel='rbf', gamma= 0.2, nu=0.01)

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

        '''
        df_train_normal = sequences_df[sequences_df['Temp_label'] == 0][['features', 'Temp_label', 'Label']].copy()
        df_train_normal['Final_Label'] = df_train_normal['Label']
        df_train_normal = df_train_normal[['features', 'Temp_label', 'Label', 'Final_Label']]

        df_train_unlabeled = sequences_df[sequences_df['Temp_label'] == 999][['features', 'Temp_label', 'Label']].copy()
        df_train_unlabeled['Final_Label'] = pseudo_labels
        df_train_unlabeled = df_train_unlabeled[['features', 'Temp_label', 'Label', 'Final_Label']]

        # Combine both into a single DataFrame
        df_final = pd.concat([df_train_normal, df_train_unlabeled], ignore_index=True)

        # Final evaluation
        y_true = df_final['Label'].values
        y_pred = df_final['Final_Label'].values
        print("\nFull Training Data Classification Report (One-Class SVM):")
        print(classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"]))
        # exit()

        # Prepare test data
        df_test = sequences_df[sequences_df['Temp_label'] == 888][['features', 'Temp_label', 'Label']].copy()
        df_test = df_test[['features', 'Temp_label', 'Label']]

        df_val = sequences_df[sequences_df['Temp_label'] == 777][['features', 'Temp_label', 'Label']].copy()
        df_val = df_val[['features', 'Temp_label', 'Label']]

        # Validation check
        xx_0 = df_test[(df_test['Label'] == 0)]
        xx_1 = df_test[(df_test['Label'] == 1)]
        if len(xx_0) == 0 or len(xx_1) == 0:
            print('Error: Test data label distribution issue')
            exit()

        # Extract features and labels
        X_train = df_final['features'].tolist()
        y_train = df_final['Final_Label'].values
        X_test = df_test['features'].tolist()
        y_test_truth = df_test['Label'].values

        X_val = df_val['features'].tolist()
        y_val_truth = df_val['Label'].values
        return X_train, y_train, X_test, y_test_truth, X_val, y_val_truth
        '''
