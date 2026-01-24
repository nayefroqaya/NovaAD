import time
import warnings

import colorama
import numpy as np
import numpy as np
import numpy as np
import numpy as np
from scipy.stats import randint, uniform
from scipy.stats import randint, uniform
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
colorama.init()
GREEN = colorama.Fore.GREEN
GRAY = colorama.Fore.LIGHTBLACK_EX
RESET = colorama.Fore.RESET
YELLOW = colorama.Fore.YELLOW


class AnomalyDetector:

    @staticmethod
    def anomaly_detector(X_train, y_train, X_test, y_test_truth, X_val, y_val_truth, mode):

        print("Starting model training process...")
        start_fit = time.time()

        if mode == 'M':
            # ========================
            # 1. Base Models Initialization
            # ========================
            model_xgb_tuning = XGBClassifier(use_label_encoder=False, eval_metric='logloss',  # n_jobs=-1, cpu
                                             # tree_method="hist",  # modern tree builder
                                             device="cuda",  # ✅ GPU
                                             n_jobs=4,  # ✅ avoid CPU oversubscription during GPU training

                                             random_state=42)

            model_rf_tuning = RandomForestClassifier(n_jobs=-1, random_state=42)

            # ========================
            # 2. Hyperparameter Spaces
            # ========================
            param_dist = {'xgb': {'max_depth': [5, 7, 10], 'n_estimators': randint(100, 250),  # 500
                                  'learning_rate': uniform(0.01, 0.3), 'subsample': uniform(0.7, 0.3),
                                  'colsample_bytree': uniform(0.7, 0.3), },
                          'rf': {'max_depth': [10, 15, None], 'n_estimators': randint(100, 250),  # 500
                                 'max_features': ['sqrt', 'log2'], 'min_samples_split': [2, 5, 10]}}

            cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # 5

            # ========================
            # 3. Randomized Search – XGBoost
            # ========================
            random_search_xgb = RandomizedSearchCV(estimator=model_xgb_tuning, param_distributions=param_dist['xgb'],
                                                   n_iter=10, cv=cv_strategy, scoring='f1', n_jobs=-1, verbose=4,
                                                   random_state=42)

            random_search_xgb.fit(X_train, y_train)
            best_params_xgb = random_search_xgb.best_params_
            print("Optimal parameters for XGBoost:", best_params_xgb)

            # ========================
            # 4. Randomized Search – Random Forest
            # ========================
            random_search_rf = RandomizedSearchCV(estimator=model_rf_tuning, param_distributions=param_dist['rf'],
                                                  n_iter=10, cv=cv_strategy, scoring='f1', n_jobs=-1, verbose=4,
                                                  random_state=42)

            random_search_rf.fit(X_train, y_train)
            best_params_rf = random_search_rf.best_params_
            print("Optimal parameters for Random Forest:", best_params_rf)

            # ========================
            # 5. Final Models
            # ========================
            scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

            model_final_xgb = XGBClassifier(**best_params_xgb, scale_pos_weight=scale_pos_weight, random_state=42,
                                            # n_jobs=-1,
                                            # tree_method="hist",
                                            device="cuda",  # ✅ GPU
                                            n_jobs=4,  # ✅ keep modest when using GPU
                                            use_label_encoder=False, eval_metric='logloss')

            model_final_rf = RandomForestClassifier(**best_params_rf, class_weight="balanced", random_state=42,
                                                    n_jobs=-1)

            model_lr = LogisticRegression(class_weight="balanced")  # max_iter=500

            # ========================
            # 6. Voting Classifier
            # ========================
            voting_clf = VotingClassifier(estimators=[('xgb', model_final_xgb), ('logreg', model_lr)], voting='soft')

            # ========================
            # 7. Stacking Classifier
            # ========================
            stacking_clf = StackingClassifier(estimators=[('voting', voting_clf), ('rf2', model_final_rf), ],
                                              final_estimator=LogisticRegression())

            # ========================
            # 8. Probability Calibration
            # ========================
            calibrated_stacking = CalibratedClassifierCV(stacking_clf, method='sigmoid', cv=3)

            # ========================
            # 9. Model Training
            # ========================
            calibrated_stacking.fit(X_train, y_train)

            end_fit = time.time()
            fit_time = (end_fit - start_fit) / 60
            print(f"Model training completed in {fit_time:.2f} minutes")

            # =====================================================
            # 🔴 CHANGE #1: Threshold optimization moved to VALIDATION SET
            # =====================================================
            print("Optimizing threshold using validation set...")

            y_val_proba = calibrated_stacking.predict_proba(X_val)[:, 1]

            precisions, recalls, thresholds = precision_recall_curve(y_val_truth, y_val_proba)

            beta = 2
            f_beta = (1 + beta ** 2) * (precisions * recalls) / (beta ** 2 * precisions + recalls + 1e-6)

            best_idx = np.argmax(f_beta)
            best_threshold = thresholds[best_idx]

            print(f"Optimal threshold (validation, Fβ={beta}): {best_threshold:.3f}")
            print(f"Val Precision: {precisions[best_idx]:.3f}, "
                  f"Recall: {recalls[best_idx]:.3f}, "
                  f"F-beta: {f_beta[best_idx]:.3f}")

            # =====================================================
            # 🔴 CHANGE #2: Test set used ONLY for final evaluation
            # =====================================================
            print("Generating final test predictions...")
            start_predict = time.time()

            y_test_proba = calibrated_stacking.predict_proba(X_test)[:, 1]
            y_test_pred_adjusted = (y_test_proba >= best_threshold).astype(int)

            end_predict = time.time()
            predict_time = (end_predict - start_predict) / 60

            print(f"Prediction completed in {predict_time:.2f} minutes")

            return y_test_truth, y_test_pred_adjusted, fit_time, predict_time

        if mode == 'S':  # Single classifier
            # ========================
            # 1. Base Model Initialization
            # ========================
            model_xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', device="cuda", n_jobs=4,
                                      random_state=42)

            # ========================
            # 2. Hyperparameter Space
            # ========================
            param_dist_xgb = {'max_depth': [5, 7, 10], 'n_estimators': randint(100, 250),
                              'learning_rate': uniform(0.01, 0.3), 'subsample': uniform(0.7, 0.3),
                              'colsample_bytree': uniform(0.7, 0.3)}

            cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

            # ========================
            # 3. Randomized Search – XGBoost
            # ========================
            random_search_xgb = RandomizedSearchCV(estimator=model_xgb, param_distributions=param_dist_xgb, n_iter=10,
                                                   cv=cv_strategy, scoring='f1', n_jobs=-1, verbose=4, random_state=42)
            random_search_xgb.fit(X_train, y_train)
            best_params_xgb = random_search_xgb.best_params_
            print("Optimal parameters for XGBoost:", best_params_xgb)

            # ========================
            # 4. Final Model
            # ========================
            scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

            model_final = XGBClassifier(**best_params_xgb, scale_pos_weight=scale_pos_weight, random_state=42,
                                        use_label_encoder=False, eval_metric='logloss', device="cuda", n_jobs=4)

            # ========================
            # 5. Train Final Model
            # ========================
            start_fit = time.time()
            model_final.fit(X_train, y_train)
            end_fit = time.time()
            fit_time = (end_fit - start_fit) / 60
            print(f"Model training completed in {fit_time:.2f} minutes")

            # ========================
            # 6. Threshold Optimization on Validation Set
            # ========================
            y_val_proba = model_final.predict_proba(X_val)[:, 1]

            precisions, recalls, thresholds = precision_recall_curve(y_val_truth, y_val_proba)

            beta = 2
            f_beta = (1 + beta ** 2) * (precisions * recalls) / (beta ** 2 * precisions + recalls + 1e-6)
            best_idx = np.argmax(f_beta)
            best_threshold = thresholds[best_idx]

            print(f"Optimal threshold (validation, Fβ={beta}): {best_threshold:.3f}")
            print(f"Val Precision: {precisions[best_idx]:.3f}, "
                  f"Recall: {recalls[best_idx]:.3f}, "
                  f"F-beta: {f_beta[best_idx]:.3f}")

            # ========================
            # 7. Test Predictions
            # ========================
            start_predict = time.time()
            y_test_proba = model_final.predict_proba(X_test)[:, 1]
            y_test_pred_adjusted = (y_test_proba >= best_threshold).astype(int)
            end_predict = time.time()
            predict_time = (end_predict - start_predict) / 60

            print(f"Prediction completed in {predict_time:.2f} minutes")

            return y_test_truth, y_test_pred_adjusted, fit_time, predict_time
