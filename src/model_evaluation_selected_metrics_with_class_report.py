import os
import json
import pandas as pd
import numpy as np

from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)


class ModelEvaluation:

    @staticmethod
    def _normalize_binary_labels(labels):
        """
        Convert labels to binary:
            0 = normal
            1 = anomaly
        """
        y = []
        for label in labels:
            if isinstance(label, (list, tuple, np.ndarray)):
                label = np.max(label)

            if label in [1, "1", "Anomaly", "anomaly", "Abnormal", "abnormal", True]:
                y.append(1)
            else:
                y.append(0)

        return np.asarray(y, dtype=int)

    @staticmethod
    def _calculate_early_detection_metrics(
        y_true,
        y_pred,
        X_test,
        detection_steps=None,
        sequence_lengths=None,
    ):


        y_true = np.asarray(y_true, dtype=int)
        y_pred = np.asarray(y_pred, dtype=int)

        anomaly_mask = y_true == 1
        detected_mask = (y_true == 1) & (y_pred == 1)

        total_anomalies = int(np.sum(anomaly_mask))
        detected_anomalies = int(np.sum(detected_mask))

        detection_coverage = (
            detected_anomalies / total_anomalies
            if total_anomalies > 0
            else 0.0
        )

        # ------------------------------------------------------------
        # Case 1: real early-detection steps are provided
        # ------------------------------------------------------------
        if detection_steps is not None and sequence_lengths is not None:
            detection_steps = np.asarray(detection_steps, dtype=float)
            sequence_lengths = np.asarray(sequence_lengths, dtype=float)

            valid_len = min(len(y_true), len(y_pred), len(detection_steps), len(sequence_lengths))
            y_true = y_true[:valid_len]
            y_pred = y_pred[:valid_len]
            detection_steps = detection_steps[:valid_len]
            sequence_lengths = sequence_lengths[:valid_len]

            anomaly_mask = y_true == 1
            detected_mask = (y_true == 1) & (y_pred == 1)

            detected_steps = detection_steps[detected_mask]
            detected_lengths = sequence_lengths[detected_mask]
            detected_lengths = np.maximum(detected_lengths, 1.0)

            if detected_anomalies > 0:
                detected_ratios = detected_steps / detected_lengths
                avg_detection_step = float(np.mean(detected_steps))
                avg_detection_ratio = float(np.mean(detected_ratios))
            else:
                detected_ratios = np.asarray([], dtype=float)
                avg_detection_step = 0.0
                avg_detection_ratio = 0.0

            # EDR@k = percentage of all anomalies detected before k% of the sequence
            if total_anomalies > 0:
                edr_25 = float(np.sum(detected_ratios <= 0.25) / total_anomalies)
                edr_50 = float(np.sum(detected_ratios <= 0.50) / total_anomalies)
                edr_75 = float(np.sum(detected_ratios <= 0.75) / total_anomalies)
            else:
                edr_25 = 0.0
                edr_50 = 0.0
                edr_75 = 0.0

        # ------------------------------------------------------------
        # Case 2: static/full-sequence classifier fallback
        # ------------------------------------------------------------
        else:
            if hasattr(X_test, "shape") and len(X_test.shape) > 1:
                avg_detection_step = float(X_test.shape[1])
            else:
                avg_detection_step = 1.0

            if detected_anomalies > 0:
                avg_detection_ratio = 1.0
            else:
                avg_detection_ratio = 0.0

            # Static/full-sequence models detect at the end, so not early.
            edr_25 = 0.0
            edr_50 = 0.0
            edr_75 = 0.0

        return {
            "Total anomalies": int(total_anomalies),
            "Detected anomalies": int(detected_anomalies),
            "Detection coverage": float(detection_coverage),
            "Avg detection step": float(avg_detection_step),
            "Avg detection ratio": float(avg_detection_ratio),
            "EDR@25": float(edr_25),
            "EDR@50": float(edr_50),
            "EDR@75": float(edr_75),
        }

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
        y_test_score=None,  # kept for compatibility; not printed in the selected report
        dataset_root="NovaADLS/datasets",
        method_name="StackingClassifier",
        fp_unit_cost=10.0,
        fn_unit_cost=20.0,
        delay_unit_cost=5.0,
        detection_steps=None,
        sequence_lengths=None,
    ):


        # ========================
        # 0. Output directory
        # ========================
        output_dir = os.path.join(dataset_root, DATASET)
        os.makedirs(output_dir, exist_ok=True)

        y_test_truth = ModelEvaluation._normalize_binary_labels(y_test_truth)
        y_test_pred = ModelEvaluation._normalize_binary_labels(y_test_pred)

        # Make sure both arrays have the same length
        valid_len = min(len(y_test_truth), len(y_test_pred))
        y_test_truth = y_test_truth[:valid_len]
        y_test_pred = y_test_pred[:valid_len]

        # ========================
        # 1. Classification report
        # ========================
        class_report_text = classification_report(
            y_test_truth,
            y_test_pred,
            labels=[0, 1],
            target_names=["Normal (0)", "Anomaly (1)"],
            digits=4,
            zero_division=0,
        )

        class_report_dict = classification_report(
            y_test_truth,
            y_test_pred,
            labels=[0, 1],
            target_names=["Normal (0)", "Anomaly (1)"],
            digits=4,
            zero_division=0,
            output_dict=True,
        )

        print(f"\n{method_name} - Classification Report:")
        print(class_report_text)

        # ========================
        # 2. Confusion matrix
        # ========================
        cm = confusion_matrix(y_test_truth, y_test_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        # ========================
        # 3. Selected classification metrics
        #    These are anomaly-class metrics because pos_label=1 by default.
        # ========================
        precision = precision_score(y_test_truth, y_test_pred, zero_division=0)
        recall = recall_score(y_test_truth, y_test_pred, zero_division=0)
        f1 = f1_score(y_test_truth, y_test_pred, zero_division=0)

        # ========================
        # 4. Selected early-detection metrics
        # ========================
        early_metrics = ModelEvaluation._calculate_early_detection_metrics(
            y_true=y_test_truth,
            y_pred=y_test_pred,
            X_test=X_test,
            detection_steps=detection_steps,
            sequence_lengths=sequence_lengths,
        )

        detected_anomalies = early_metrics["Detected anomalies"]

        # ========================
        # 5. Selected cost-sensitive metrics
        # ========================
        false_positive_cost = float(fp * fp_unit_cost)
        false_negative_cost = float(fn * fn_unit_cost)

        # If there is no real detection timing, detected anomalies are treated
        # as detected at the full sequence, with avg ratio = 1.0 for static models.
        delay_cost = float(
            detected_anomalies
            * delay_unit_cost
            * early_metrics["Avg detection ratio"]
        )

        # ========================
        # 6. Store only selected metrics
        # ========================
        metrics = {
            "Dataset": DATASET,
            "Round": Round,
            "Method": method_name,
            "Number of sequences": int(len(y_test_truth)),

            # Selected classification metrics for anomaly class
            "Precision": float(precision),
            "Recall / TPR": float(recall),
            "F1-score": float(f1),

            # Full classification report for class 0 and class 1
            "Classification report": class_report_dict,

            # Confusion matrix values are kept because they explain the costs
            "TP": int(tp),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "Confusion Matrix": cm.tolist(),

            # Early-detection metrics
            **early_metrics,

            # Cost-sensitive metrics
            "FP unit cost": float(fp_unit_cost),
            "FN unit cost": float(fn_unit_cost),
            "Delay unit cost": float(delay_unit_cost),
            "False-positive cost": float(false_positive_cost),
            "False-negative cost": float(false_negative_cost),
            "Delay cost": float(delay_cost),
        }

        metrics_df = pd.DataFrame([metrics])

        # ========================
        # 7. Save TXT report only
        # ========================
        safe_method_name = str(method_name).replace(" ", "_")
        metrics_txt = os.path.join(
            output_dir,
            f"{Round}_{DATASET}_{safe_method_name}_selected_metrics.txt"
        )

        with open(metrics_txt, "w", encoding="utf-8") as f:
            f.write(f"{DATASET} - {Round} - {method_name} Selected Evaluation Metrics\n")
            f.write("=" * 80 + "\n\n")

            f.write("[Classification Metrics - Anomaly Class]\n")
            f.write("Positive class        : 1 = anomaly\n")
            f.write(f"Precision             : {precision:.4f}\n")
            f.write(f"Recall / TPR          : {recall:.4f}\n")
            f.write(f"F1-score              : {f1:.4f}\n\n")

            f.write("[Classification Report - Class 0 and Class 1]\n")
            f.write("Class 0               : Normal\n")
            f.write("Class 1               : Anomaly\n\n")
            f.write(class_report_text)
            f.write("\n")

            f.write("[Confusion Matrix]\n")
            f.write("Labels: 0=normal, 1=anomaly\n")
            f.write(str(cm.tolist()) + "\n")
            f.write(f"TP={tp} TN={tn} FP={fp} FN={fn}\n\n")

            f.write("[Early Detection Metrics]\n")
            f.write(f"Total anomalies       : {early_metrics['Total anomalies']}\n")
            f.write(f"Detected anomalies    : {early_metrics['Detected anomalies']}\n")
            f.write(f"Detection coverage    : {early_metrics['Detection coverage']:.4f}\n")
            f.write(f"Avg detection step    : {early_metrics['Avg detection step']:.4f}\n")
            f.write(f"Avg detection ratio   : {early_metrics['Avg detection ratio']:.4f}\n")
            f.write(f"EDR@25                : {early_metrics['EDR@25']:.4f}\n")
            f.write(f"EDR@50                : {early_metrics['EDR@50']:.4f}\n")
            f.write(f"EDR@75                : {early_metrics['EDR@75']:.4f}\n\n")

            f.write("[Cost-Sensitive Metrics]\n")
            f.write(f"False-positive cost   : {false_positive_cost:.4f}\n")
            f.write(f"False-negative cost   : {false_negative_cost:.4f}\n")
            f.write(f"Delay cost            : {delay_cost:.4f}\n")

        print(f"Selected metrics TXT saved to: {metrics_txt}")

        # ========================
        # 8. Feature names
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
            X_train = pd.DataFrame(
                X_train,
                columns=feature_names[: X_train.shape[1]]
            )

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
            f"{Round}_{DATASET}_{safe_method_name}_mi_scores.csv"
        )

        mi_df.to_csv(mi_output_filename, index=False)

        print(f"Mutual Information scores saved to: {mi_output_filename}")

        return mi_df, metrics_df
