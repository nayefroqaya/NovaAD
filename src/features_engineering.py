import warnings
from itertools import product

import colorama
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
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

       #Define safe aggregation function
        def safe_stack(feature_list):
          arrays = [np.asarray(f, dtype=np.float32) for f in feature_list if f is not None and len(f) > 0]
          if len(arrays) == 0:

              first = feature_list.iloc[0] if len(feature_list) > 0 else np.zeros(51, dtype=np.float32)
              return np.zeros(len(first), dtype=np.float32) if first is not None else np.zeros(51, dtype=np.float32)

              # Return zeros with the same length as any feature vector
#              return np.zeros(len(feature_list[0]), dtype=np.float32)
          return np.mean(np.stack(arrays), axis=0)

        log_normal_labelled = df_train_with_test_with_val[df_train_with_test_with_val['Temp_label'] == 0].copy()
        # 2️⃣ Remove UNKNOWN blocks (case-insensitive)
        log_normal_labelled = log_normal_labelled[
                ~log_normal_labelled['Node_block_id'].astype(str).str.contains("UNKNOWN", case=False, na=False)
        ].reset_index(drop=True)

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
        summed_df_normal_labelled_train = summed_df_normal_labelled_train.merge(sequence_labels_normal_labelled, on='Node_block_id')
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
        summed_df_combine_unlabelled_train = log_remain_normal_anomaly_unlabelled.groupby('Node_block_id')['features'].apply(
           safe_stack).reset_index()

       # 5️⃣ Assign sequence labels (normal/anomaly) safely
        sequence_labels_combine_unlabelled  = log_remain_normal_anomaly_unlabelled.groupby('Node_block_id')['Label'].apply(
           lambda x: 'anomaly' if any(lbl != 'Normal' for lbl in x) else 'normal').reset_index()

       # 6️⃣ Merge features with labeel 
        # Merging the summed feature vectors with their respective labels
        summed_df_combine_unlabelled_train = summed_df_combine_unlabelled_train.merge(
            sequence_labels_combine_unlabelled,
            on='Node_block_id')
        summed_df_combine_unlabelled_train['Temp_label'] = 999  # Normal and anomaly unlabeled

        # Validation checks
        xx_anomaly = summed_df_combine_unlabelled_train[(summed_df_combine_unlabelled_train['Label'] == 'anomaly')]
        yy_normal = summed_df_combine_unlabelled_train[(summed_df_combine_unlabelled_train['Label'] == 'normal')]

        if len(xx_anomaly) == 0 or len(yy_normal) == 0:
            print('Error in summing_Train_test_together function: Unlabeled data validation failed')
            exit()

        summed_df_train = pd.concat([summed_df_normal_labelled_train, summed_df_combine_unlabelled_train])

        # (3) Test dataset: ------------------------------------------------------------------------------------
        summed_df_combine_unlabelled_test = log_test_unlabelled.groupby('Node_block_id')['features'].apply(
           safe_stack).reset_index()
        sequence_labels_combine_unlabelled_test = log_test_unlabelled.groupby('Node_block_id')[
                'Label'].apply(lambda x: 'anomaly' if any(lbl != 'Normal' for lbl in x) else 'normal').reset_index()
        # Merging the summed feature vectors with their respective labels
        summed_df_combine_unlabelled_test = summed_df_combine_unlabelled_test.merge(
            sequence_labels_combine_unlabelled_test,
            on='Node_block_id')
        summed_df_combine_unlabelled_test['Temp_label'] = 888  # Test normal and anomaly unlabeled
        summed_df_test = summed_df_combine_unlabelled_test

        # (4) Val dataset: ------------------------------------------------------------------------------------
        summed_df_val_dataset= log_val_labeled.groupby('Node_block_id')['features'].apply(
           safe_stack).reset_index()
        sequence_labels_val = log_val_labeled.groupby('Node_block_id')[
                'Label'].apply(lambda x: 'anomaly' if any(lbl != 'Normal' for lbl in x) else 'normal').reset_index()
        # Merging the summed feature vectors with their respective labels
        summed_df_val = summed_df_val_dataset.merge(
            sequence_labels_val,
            on='Node_block_id')
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
        feature_columns = ['sentiment_label', 'Dominant_Topic', 'num_words', 'Character_Count',
                           'entropy','year', 'month', 'day', 'hour', 'minute', 'second']

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
        final_train_with_test_with_val = final_train_with_test_with_val.astype(
            {col: 'float32' if final_train_with_test_with_val[col].dtype == 'float64'
            else 'int32' if final_train_with_test_with_val[col].dtype == 'int64'
            else final_train_with_test_with_val[col].dtype for col in final_train_with_test_with_val.columns}
        )

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

    @staticmethod
    def novelty_detection_label_establishment(sequences_df, X_train_normal_labelled, X_unlabeled_train,
                                              ground_truth_unlabeled_data_from_train):
        # Parameter grid for One-Class SVM
        gamma_values = np.linspace(0.2, 0.5, 5)
        nu_values = np.linspace(0.01, 0.08, 5)

        best_params = None
        best_score = -np.inf  # Higher LOF separation score is better
        '''
        # Grid Search Over Gamma & Nu
        for gamma, nu in product(gamma_values, nu_values):
            print(f'Testing parameters: Gamma={gamma}, Nu={nu}')

            # Train One-Class SVM ONLY on Normal Labeled Data
            oc_svm = OneClassSVM(kernel="rbf", gamma=gamma, nu=nu)
            oc_svm.fit(X_train_normal_labelled)

            # Predict Labels on Unlabeled Data (1 = normal, -1 = anomaly)
            preds = oc_svm.predict(X_unlabeled_train)

            # Compute LOF Scores (Only for Points Classified as Normal)
            lof = LocalOutlierFactor(n_neighbors=20)
            lof_scores = -lof.fit_predict(X_unlabeled_train[preds == 1])  # Higher values = anomalies

            # Compute Silhouette Score
            if len(set(preds)) > 1:
                silhouette = silhouette_score(X_unlabeled_train, preds)
            else:
                silhouette = -1  # Invalid case (all one class)

            # Hybrid Metric = Weighted Sum of LOF & Silhouette
            hybrid_score = 0.5 * np.mean(lof_scores) + 0.5 * silhouette

            # Select Best Gamma & Nu
            if hybrid_score > best_score:
                best_score = hybrid_score
                best_params = (gamma, nu)

        print(f"Optimal parameters found: Gamma={best_params[0]}, Nu={best_params[1]}")
        '''
        #exit()


        
        # Train final model with optimal parameters
        #oc_svm = OneClassSVM(kernel='rbf', gamma=best_params[0], nu=best_params[1])
        oc_svm = OneClassSVM(kernel='rbf', gamma= 0.425, nu=0.01)

        oc_svm.fit(X_train_normal_labelled)

        # Make predictions
        y_pred = oc_svm.predict(X_unlabeled_train)
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
        df_train_normal = sequences_df[sequences_df['Temp_label'] == 0][['features', 'Temp_label', 'Label']].copy()
        df_train_normal['Final_Label'] = df_train_normal['Label']
        df_train_normal = df_train_normal[['features', 'Temp_label', 'Label', 'Final_Label']]

        df_train_unlabeled = sequences_df[sequences_df['Temp_label'] == 999][
            ['features', 'Temp_label', 'Label']].copy()
        df_train_unlabeled['Final_Label'] = pseudo_labels
        df_train_unlabeled = df_train_unlabeled[['features', 'Temp_label', 'Label', 'Final_Label']]

        # Combine both into a single DataFrame
        df_final = pd.concat([df_train_normal, df_train_unlabeled], ignore_index=True)

        # Final evaluation
        y_true = df_final['Label'].values
        y_pred = df_final['Final_Label'].values
        print("\nFull Training Data Classification Report (One-Class SVM):")
        print(classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"]))

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

