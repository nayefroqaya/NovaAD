import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import classification_report
import numpy as np


class ModelEvaluation:
    """Class for evaluating model performance and feature importance."""

    @staticmethod
    def evaluation( Round,X_train, y_train ,
        number_component, y_test_truth, y_test_pred, DATASET, X_test):
        """
        Evaluate model performance and compute feature importance metrics.

        Parameters:
        number_components (int): Number of principal components used
        y_test_truth (array): Ground truth labels
        y_test_pred (array): Predicted labels
        dataset (str): Dataset identifier for output file naming
        X_test (DataFrame or array): Test features
        """

        # Print classification report
        print("\n StackingClassifier - Classification Report:")
        print(classification_report(y_test_truth, y_test_pred,
                                    target_names=["Class 0", "Class 1"], digits=3))
        # ========================
        # 2. Feature Names
        # ========================
        feature_names = (
                [f"component_{i + 1}" for i in range(number_component)] + ['sentiment', 'Dominant_Topic', 'word_count',
            'character_count', 'entropy', 'year', 'month', 'day', 'hour', 'minute', 'second'])

        # ========================
        # 3. Ensure X_train is DataFrame
        # ========================
        if not isinstance(X_train, pd.DataFrame):
            #X_train = pd.DataFrame(X_train, columns=feature_names[:X_train.shape[1]])
            X_train = np.asarray(X_train)

            X_train = pd.DataFrame(X_train, columns=feature_names[:X_train.shape[1]])

        # ========================
        # 4. Mutual Information (TRAIN DATA ONLY) ✅
        # ========================
        print("Calculating feature importance using Mutual Information (training data)...")

        mi_scores = mutual_info_classif(X_train, y_train, random_state=42)

        # ========================
        # 5. Create MI DataFrame
        # ========================
        mi_df = pd.DataFrame({"Feature": X_train.columns, "MI_Score": mi_scores}).sort_values(by="MI_Score",
                                                                                              ascending=False)

        # ========================
        # 6. Save Results
        # ========================
        output_filename = f"{Round}_{DATASET}_evaluation_mi_scores.csv"
        mi_df.to_csv(output_filename, index=False)

        print(f"Mutual Information scores saved to: {output_filename}")

        return mi_df