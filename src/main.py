import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # ⛔ Disable GPU completely
import warnings
import colorama
import pandas as pd
import torch
import os

from utility import Utilities
from anomaly_detection import AnomalyDetector
from features_engineering import FeaturesEngineering
from features_extracting import FeaturesExtractor
from logdata_read import LogdataRead
from model_evaluation import ModelEvaluation

# ====================== Setup ======================
warnings.filterwarnings('ignore')
colorama.init()

GREEN = colorama.Fore.GREEN
GRAY = colorama.Fore.LIGHTBLACK_EX
RESET = colorama.Fore.RESET
YELLOW = colorama.Fore.YELLOW


# ====================== Main ======================
def main():
    # ---------------- Device check ------------------
    #if torch.cuda.is_available():
    #    print(f"{GREEN}GPU detected. Using GPU for encoding.{RESET}")
    #else:
    #    print(f"{YELLOW}No GPU detected. Using CPU for encoding. Exiting program.{RESET}")
    #    exit()
    # ---------------- Device setup (CPU ONLY) ----------------
    device = torch.device("cpu")
    torch.backends.cudnn.enabled = False
    torch.backends.cuda.enabled = False
    print(f"{YELLOW}Using device: CPU only{RESET}")
    print(f"{YELLOW}Using device: CPU (GPU disabled){RESET}")
    # ---------------- Display options ----------------
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    # ---------------- Project configuration ----------------
#<<<<<<< HEAD
#    DATASET = 'BGL'
#=======
    DATASET = 'BGL'
#>>>>>>> 42c37e9 (update input name)
    DATASETS_FOLDER = 'datasets'
    Round= '1'
    mode='M'  # M multi classifier - S single classifier


    # Paths
    ALL_DATASET_LOG_PATH = f'../{DATASETS_FOLDER}/{DATASET}/{DATASET}.LOG'
    ALL_DATASET_CSV_PATH = f'../{DATASETS_FOLDER}/{DATASET}/{DATASET}.csv'
    DOC_TOPIC_DF_PATH = f'../{DATASETS_FOLDER}/{DATASET}/{DATASET}_All_doc_topic_df.pkl'
    SENTIMENT_DF_PATH = f'../{DATASETS_FOLDER}/{DATASET}/{DATASET}_All_sentiment_df.pkl'
    PRE_FINAL_GLOBAL_FEATURES_PKL_PATH = f'../{DATASETS_FOLDER}/{DATASET}/{DATASET}_All_pre_final_global_features.pkl'

    # ---------------- Initialize classes ----------------
    logdata_read_obj = LogdataRead()
    features_extracting_obj = FeaturesExtractor()
    features_engineering_obj = FeaturesEngineering()
    anomaly_detection_obj = AnomalyDetector()
    model_evaluation_obj = ModelEvaluation()
    utilities_obj = Utilities()

    # ---------------- Data as CSV ----------------
#<<<<<<< HEAD
    #logdata_read_obj.read_original_data_log_from_log_to_csv(DATASET, ALL_DATASET_CSV_PATH)
    #print(' Reading the file was done successfully ')
    #exit()
#=======
    #logdata_read_obj.read_original_data_log_from_log_to_csv(DATASET, ALL_DATASET_CSV_PATH)
    #print(' Reading the file was done successfully ')
    #exit()
#>>>>>>> 42c37e9 (update input name)

    # ---------------- Dataset Splitting ----------------
    #print(f"{GRAY}Splitting dataset into training, validation, and test sets...{RESET}")
    #train_df, validate_df, test_df, df_features = utilities_obj.dataset_splitting(
    #    ALL_DATASET_CSV_PATH, DATASET, Round
    #)
    #exit()
    # ---------------- Process normal data ----------------
    print(f"{GRAY}Processing normal data portion in the dataset...{RESET}")
    save_path = os.path.join(f"../datasets/{DATASET}", f"{Round}_{DATASET}_Splitted_Datasets")
    train_df = pd.read_pickle(os.path.join(save_path, "train_df.pkl"))
    val_df = pd.read_pickle(os.path.join(save_path, "val_df.pkl"))
    test_df = pd.read_pickle(os.path.join(save_path, "test_df.pkl"))

    final_train_with_test_with_val = utilities_obj.processing_data_portion(
        train_df, val_df, test_df )
    #exit()

    # ---------------- Features Extracting ----------------
    print(f"{GRAY}Extracting features for training and test datasets...{RESET}")
    number_component, best_topic_number = features_extracting_obj.features_extracting_configuring_tuning(
        features_extracting_obj,
        DOC_TOPIC_DF_PATH,
        SENTIMENT_DF_PATH,
        DATASET,
        PRE_FINAL_GLOBAL_FEATURES_PKL_PATH,
        final_train_with_test_with_val
    )
    final_train_with_test_with_val = pd.read_pickle(PRE_FINAL_GLOBAL_FEATURES_PKL_PATH)
    # ---------------- Features Engineering: Aggregation/Transformation ----------------
    print(f"{GRAY}Aggregating and transforming features...{RESET}")
    sequences_df, x_sequences_df, y_sequences_df = features_engineering_obj.features_aggregation_transformation(
        final_train_with_test_with_val,
        DATASET
    )
    # ---------------- Prepare datasets ----------------
    print(f"{GRAY}Preparing training and evaluation datasets...{RESET}")
    x_train_normal_labelled = x_sequences_df[y_sequences_df == 0]
    X_train_all_data = x_sequences_df[y_sequences_df != 888]
    x_unlabeled_from_train = x_sequences_df[y_sequences_df == 999]

    labelled_df_from_train = sequences_df[sequences_df['Temp_label'] == 0]
    ground_truth_labeled_data_from_train = labelled_df_from_train['Label']

    labelled_df_from_train_all_data = sequences_df[sequences_df['Temp_label'] != 888]
    ground_truth_train_all_data = labelled_df_from_train_all_data['Label']

    unlabeled_df_from_train = sequences_df[sequences_df['Temp_label'] == 999]
    ground_truth_unlabeled_data_from_train = unlabeled_df_from_train['Label']

    unlabeled_df_from_test = sequences_df[sequences_df['Temp_label'] == 888]
    ground_truth_unlabeled_data_from_test = unlabeled_df_from_test['Label']

    labeled_df_from_val = sequences_df[sequences_df['Temp_label'] == 777]
    ground_truth_labeled_data_from_val = labeled_df_from_val['Label']

    # ---------------- Novelty detection and label establishment ----------------
    print(f"{GRAY}Performing novelty detection and establishing labels...{RESET}")
    X_train, y_train, X_test, y_test_truth, X_val, y_val_truth = features_engineering_obj.novelty_detection_label_establishment(
        sequences_df,
        x_train_normal_labelled,
        x_unlabeled_from_train,
        ground_truth_unlabeled_data_from_train
    )
    # ---------------- Anomaly Detection ----------------
    print(f"{GRAY}Running anomaly detection on test dataset...{RESET}")
    y_test_truth, y_test_pred, fit_time, predict_time = anomaly_detection_obj.anomaly_detector(X_train, y_train, X_test,
        y_test_truth, X_val, y_val_truth, mode)

    # ---------------- Model Evaluation ----------------
    print(f"{GRAY}Evaluating model performance...{RESET}")
    model_evaluation_obj.evaluation( Round,X_train, y_train ,
        number_component, y_test_truth, y_test_pred, DATASET, X_test
    )

    print(f"Model training completed in {fit_time:.2f} minutes")
    print(f"Prediction completed in {predict_time:.2f} minutes")


if __name__ == "__main__":
    main()
