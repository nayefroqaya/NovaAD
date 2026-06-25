import os
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
)
import numpy as np
import json


class ModelEvaluation:
    """Class for evaluating model performance and feature importance."""

    @staticmethod
    def evaluation(
        Round,
        X_train,
        y_train,
        number_component,
        y_test_truth,
        y_test_pred,
        DATASET,
        X_test,
        y_test_score=None,
        dataset_root="datasets",
        method_name="StackingClassifier",
        fp_unit_cost=10.0,
        fn_unit_cost=20.0,
        delay_unit_cost=5.0,
    ):
        """
        Evaluate model performance and save results inside each dataset folder.

        Results will be saved to:
            datasets/{DATASET}/StackingClassifier_results/
        """

        # ========================
        # 0. Output path
        # ========================
        output_dir = os.path.join(dataset_root, DATASET, f"{method_name}_results")
        os.makedirs(output_dir, exist_ok=True)

        y_test_truth = np.asarray(y_test_truth)
        y_test_pred = np.asarray(y_test_pred)

        # ========================
        # 1. Classification Report
        # ========================
        print(f"\n{method_name} - Classification Report:")
        print(
            classification_report(
                y_test_truth,
                y_test_pred,
                target_names=["Normal", "Anomaly"],
                digits=4,
                zero_division=0,
            )
        )

        # ========================
        # 2. Confusion Matrix
        # ========================
        cm = confusion_matrix(y_test_truth, y_test_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        total_sequences = len(y_test_truth)
        total_anomalies = int(tp + fn)
        detected_anomalies = int(tp)

        # ========================
        # 3. Classification Metrics
        # ========================
        accuracy = accuracy_score(y_test_truth, y_test_pred)
        balanced_accuracy = balanced_accuracy_score(y_test_truth, y_test_pred)

        precision = precision_score(y_test_truth, y_test_pred, zero_division=0)
        recall = recall_score(y_test_truth, y_test_pred, zero_division=0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        f1 = f1_score(y_test_truth, y_test_pred, zero_division=0)
        f2 = fbeta_score(y_test_truth, y_test_pred, beta=2, zero_division=0)

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        mcc = matthews_corrcoef(y_test_truth, y_test_pred)

        # AUROC / AUPRC need anomaly scores, not only labels
        if y_test_score is not None:
            y_test_score = np.asarray(y_test_score)
            try:
                auroc = roc_auc_score(y_test_truth, y_test_score)
                auprc = average_precision_score(y_test_truth, y_test_score)
            except Exception:
                auroc = None
                auprc = None
        else:
            auroc = None
            auprc = None

        # ========================
        # 4. Early Detection Metrics
        # Static models predict after the full sequence
        # ========================
        detection_coverage = recall

        avg_detection_step = X_test.shape[1] if hasattr(X_test, "shape") and len(X_test.shape) > 1 else 1
        avg_detection_ratio = 1.0 if detected_anomalies > 0 else 0.0
        median_detection_ratio = 1.0 if detected_anomalies > 0 else 0.0

        edr_25 = 0.0
        edr_50 = 0.0
        edr_75 = 0.0

        # ========================
        # 5. RL Behavior Metrics
        # ========================
        average_reward = None
        alert_rate = np.mean(y_test_pred == 1)

        # ========================
        # 6. Cost Metrics
        # ========================
        fp_total_cost = fp * fp_unit_cost
        fn_total_cost = fn * fn_unit_cost
        delay_total_cost = detected_anomalies * delay_unit_cost * avg_detection_ratio

        total_cost = fp_total_cost + fn_total_cost + delay_total_cost
        avg_cost_per_sequence = total_cost / total_sequences if total_sequences > 0 else 0.0

        # ========================
        # 7. Save Metrics
        # ========================
        metrics = {
            "Dataset": DATASET,
            "Round": Round,
            "Method": method_name,
            "Number of sequences": int(total_sequences),

            "Accuracy": float(accuracy),
            "Balanced Accuracy": float(balanced_accuracy),
            "Precision": float(precision),
            "Recall / TPR": float(recall),
            "Specificity / TNR": float(specificity),
            "F1-score": float(f1),
            "F2-score": float(f2),
            "FPR": float(fpr),
            "FNR": float(fnr),
            "MCC": float(mcc),
            "AUROC": None if auroc is None else float(auroc),
            "AUPRC": None if auprc is None else float(auprc),

            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
            "Confusion Matrix": cm.tolist(),

            "Total anomalies": int(total_anomalies),
            "Detected anomalies": int(detected_anomalies),
            "Detection coverage": float(detection_coverage),
            "Avg detection step": float(avg_detection_step),
            "Avg detection ratio": float(avg_detection_ratio),
            "Median detection ratio": float(median_detection_ratio),
            "EDR@25": float(edr_25),
            "EDR@50": float(edr_50),
            "EDR@75": float(edr_75),

            "Average reward": average_reward,
            "Alert rate": float(alert_rate),

            "FP unit cost": float(fp_unit_cost),
            "FN unit cost": float(fn_unit_cost),
            "Delay unit cost": float(delay_unit_cost),
            "FP total cost": float(fp_total_cost),
            "FN total cost": float(fn_total_cost),
            "Delay total cost": float(delay_total_cost),
            "Total cost": float(total_cost),
            "Avg cost / sequence": float(avg_cost_per_sequence),
        }

        metrics_df = pd.DataFrame([metrics])

        metrics_csv = os.path.join(output_dir, f"{Round}_{DATASET}_{method_name}_metrics.csv")
        metrics_json = os.path.join(output_dir, f"{Round}_{DATASET}_{method_name}_metrics.json")
        metrics_txt = os.path.join(output_dir, f"{Round}_{DATASET}_{method_name}_metrics.txt")

        metrics_df.to_csv(metrics_csv, index=False)

        with open(metrics_json, "w") as f:
            json.dump(metrics, f, indent=4)

        with open(metrics_txt, "w") as f:
            f.write(f"{DATASET} - {Round} - {method_name} Evaluation Metrics\n")
            f.write("=" * 70 + "\n\n")

            f.write("[Classification Metrics]\n")
            f.write(f"Number of sequences   : {total_sequences}\n")
            f.write(f"Accuracy              : {accuracy:.4f}\n")
            f.write(f"Balanced Accuracy     : {balanced_accuracy:.4f}\n")
            f.write(f"Precision             : {precision:.4f}\n")
            f.write(f"Recall / TPR          : {recall:.4f}\n")
            f.write(f"Specificity / TNR     : {specificity:.4f}\n")
            f.write(f"F1-score              : {f1:.4f}\n")
            f.write(f"F2-score              : {f2:.4f}\n")
            f.write(f"FPR                   : {fpr:.4f}\n")
            f.write(f"FNR                   : {fnr:.4f}\n")
            f.write(f"MCC                   : {mcc:.4f}\n")
            f.write(f"AUROC                 : {'N/A' if auroc is None else f'{auroc:.4f}'}\n")
            f.write(f"AUPRC                 : {'N/A' if auprc is None else f'{auprc:.4f}'}\n")
            f.write(f"Confusion Matrix      : {cm.tolist()}\n")
            f.write(f"TP={tp} TN={tn} FP={fp} FN={fn}\n\n")

            f.write("[Early Detection Metrics]\n")
            f.write(f"Total anomalies       : {total_anomalies}\n")
            f.write(f"Detected anomalies    : {detected_anomalies}\n")
            f.write(f"Detection coverage    : {detection_coverage:.4f}\n")
            f.write(f"Avg detection step    : {avg_detection_step:.4f}\n")
            f.write(f"Avg detection ratio   : {avg_detection_ratio:.4f}\n")
            f.write(f"Median detect. ratio  : {median_detection_ratio:.4f}\n")
            f.write(f"EDR@25                : {edr_25:.4f}\n")
            f.write(f"EDR@50                : {edr_50:.4f}\n")
            f.write(f"EDR@75                : {edr_75:.4f}\n\n")

            f.write("[RL Behavior Metrics]\n")
            f.write("Average reward        : N/A\n")
            f.write(f"Alert rate            : {alert_rate:.4f}\n\n")

            f.write("[Cost Metrics]\n")
            f.write(f"FP unit cost          : {fp_unit_cost:.4f}\n")
            f.write(f"FN unit cost          : {fn_unit_cost:.4f}\n")
            f.write(f"Delay unit cost       : {delay_unit_cost:.4f}\n")
            f.write(f"FP total cost         : {fp_total_cost:.4f}\n")
            f.write(f"FN total cost         : {fn_total_cost:.4f}\n")
            f.write(f"Delay total cost      : {delay_total_cost:.4f}\n")
            f.write(f"Total cost            : {total_cost:.4f}\n")
            f.write(f"Avg cost / sequence   : {avg_cost_per_sequence:.4f}\n")

        print(f"Evaluation metrics saved to: {metrics_csv}")
        print(f"Evaluation metrics JSON saved to: {metrics_json}")
        print(f"Evaluation metrics TXT saved to: {metrics_txt}")

        # ========================
        # 8. Feature Names
        # ========================
        feature_names = (
            [f"component_{i + 1}" for i in range(number_component)]
            + [
                "sentiment",
                "Dominant_Topic",
                "word_count",
                "character_count",
                "entropy",
                "year",
                "month",
                "day",
                "hour",
                "minute",
                "second",
            ]
        )

        # ========================
        # 9. Ensure X_train is DataFrame
        # ========================
        if not isinstance(X_train, pd.DataFrame):
            X_train = np.asarray(X_train)
            X_train = pd.DataFrame(X_train, columns=feature_names[: X_train.shape[1]])

        # ========================
        # 10. Mutual Information
        # ========================
        print("Calculating feature importance using Mutual Information training data...")

        mi_scores = mutual_info_classif(X_train, y_train, random_state=42)

        mi_df = pd.DataFrame(
            {
                "Feature": X_train.columns,
                "MI_Score": mi_scores,
            }
        ).sort_values(by="MI_Score", ascending=False)

        mi_output_filename = os.path.join(
            output_dir,
            f"{Round}_{DATASET}_{method_name}_mi_scores.csv"
        )

        mi_df.to_csv(mi_output_filename, index=False)

        print(f"Mutual Information scores saved to: {mi_output_filename}")

        return mi_df, metrics_df