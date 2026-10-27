# RL_Early: NovaADLS Early baseline. Minimal prefix-aware extension of the original implementation.

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # ⛔ Disable GPU completely
import warnings

import colorama
import pandas as pd
import torch

from RL_Early_anomaly_detection import AnomalyDetector
from RL_Early_features_engineering import FeaturesEngineering
from RL_Early_features_extracting import FeaturesExtractor
from RL_Early_logdata_read import LogdataRead
from RL_Early_model_evaluation import ModelEvaluation
from RL_Early_utility import Utilities
import psutil
import platform

# ====================== Setup ======================
warnings.filterwarnings('ignore')
colorama.init()

GREEN = colorama.Fore.GREEN
GRAY = colorama.Fore.LIGHTBLACK_EX
RESET = colorama.Fore.RESET
YELLOW = colorama.Fore.YELLOW


# ====================== Main ======================
def main():
    '''

    # -----------------------------
    # System Info
    # -----------------------------
    vm = psutil.virtual_memory()
    total_gb = vm.total / (1024 ** 3)
    available_gb = vm.available / (1024 ** 3)
    used_gb = vm.used / (1024 ** 3)
    percent_used = vm.percent

    logical_cores = psutil.cpu_count(logical=True)
    physical_cores = psutil.cpu_count(logical=False)

    print("=== System Info ===")
    print(f"Total RAM:     {total_gb:.2f} GB")
    print(f"Available RAM: {available_gb:.2f} GB")
    print(f"Used RAM:      {used_gb:.2f} GB ({percent_used:.1f}%)")
    print(f"CPU cores (physical/logical): {physical_cores} / {logical_cores}")
    print(f"OS: {platform.platform()}")
    print(f"Python version: {platform.python_version()}")

    # -----------------------------
    # Match Spark resource logic
    # -----------------------------
    python_memory_gb = max(4, int(available_gb * 0.8))
    num_cores = logical_cores

    print(f"Target sklearn memory budget: {python_memory_gb} GB")
    print(f"Target sklearn CPU budget: {num_cores} logical cores")

    # -----------------------------
    # Thread control
    # MUST be before numpy / sklearn imports
    # -----------------------------
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["JOBLIB_TEMP_FOLDER"] = "tmp/sklearn-spill"
    '''



    # ---------------- Device check ------------------
    if torch.cuda.is_available():
        print(f"{GREEN}GPU detected. Using GPU for encoding.{RESET}")
    else:
        print(f"{YELLOW}No GPU detected. Using CPU for encoding. Exiting program.{RESET}")
    #    exit()
    # ---------------- Device setup (CPU ONLY) ----------------
    # device = torch.device("cpu")
    #    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    #   torch.backends.cudnn.enabled = False
    #   torch.backends.cuda.enabled = False
    #   print(f"{YELLOW}Using device: CPU only{RESET}")
    #   print(f"{YELLOW}Using device: CPU (GPU disabled){RESET}")
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Enable cuDNN for GPU acceleration
    #torch.backends.cudnn.enabled = True

    # ---------------- Display options ----------------
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    # ---------------- Project configuration ----------------

    DATASET = 'HDFS'
    DATASETS_FOLDER = 'datasets'
    Round = '1'
    mode = 'M'  # M multi classifier - S single classifier
    Mix_or_stable = '0'  # 0 Full stable subset  / 1 mix subset

    # Paths first paper
    ALL_DATASET_LOG_PATH = f'../{DATASETS_FOLDER}/{DATASET}/{DATASET}.LOG'

    # path for full data CSV file
    #ALL_DATASET_CSV_PATH = f'../{DATASETS_FOLDER}/{DATASET}/{DATASET}.csv' # data first paper
    ALL_DATASET_CSV_PATH = f'../../NovaAD_Plus/{DATASETS_FOLDER}/{DATASET}/{DATASET}.csv' # data second paper

    # Path features results- paper1
    #DOC_TOPIC_DF_PATH = f'../{DATASETS_FOLDER}/{DATASET}/{Round}_{DATASET}_All_doc_topic_df.pkl'
    #SENTIMENT_DF_PATH = f'../{DATASETS_FOLDER}/{DATASET}/{Round}_{DATASET}_All_sentiment_df.pkl'
    #PRE_FINAL_GLOBAL_FEATURES_PKL_PATH = f'../{DATASETS_FOLDER}/{DATASET}/{Round}_{DATASET}_All_pre_final_global_features.pkl'

    # Path features results- paper2:
    #DOC_TOPIC_DF_PATH = f'../../NovaAD_Plus/{DATASETS_FOLDER}/{DATASET}/{Round}_{DATASET}_All_doc_topic_df.pkl'
    #SENTIMENT_DF_PATH = f'../../NovaAD_Plus/{DATASETS_FOLDER}/{DATASET}/{Round}_{DATASET}_All_sentiment_df.pkl'
    #PRE_FINAL_GLOBAL_FEATURES_PKL_PATH = f'../../NovaAD_Plus/{DATASETS_FOLDER}/{DATASET}/{Round}_{DATASET}_All_pre_final_global_features.pkl'

    # Path features results- paper3:
    DOC_TOPIC_DF_PATH = f'../../LWADLS/{DATASETS_FOLDER}/{DATASET}/All_doc_topic_df.pkl'
    SENTIMENT_DF_PATH = f'../../LWADLS/{DATASETS_FOLDER}/{DATASET}/ll_sentiment_df.pkl'
    PRE_FINAL_GLOBAL_FEATURES_PKL_PATH = f'../../LWADLS/{DATASETS_FOLDER}/{DATASET}/All_pre_final_global_features.pkl'


    # ---------------- Initialize classes ----------------
    logdata_read_obj = LogdataRead()
    features_extracting_obj = FeaturesExtractor()
    features_engineering_obj = FeaturesEngineering()
    anomaly_detection_obj = AnomalyDetector()
    model_evaluation_obj = ModelEvaluation()

    utilities_obj = Utilities()

    '''
    # ---------------- Data as CSV ----------------
    #logdata_read_obj.read_original_data_log_from_log_to_csv(DATASET, ALL_DATASET_CSV_PATH)
    #print(' Reading the file was done successfully ')
    #exit()
    

    # ---------------- Dataset Splitting ----------------
    print(f"{GRAY}Splitting dataset into training, validation, and test sets...{RESET}")
    ##*train_df, validate_df, test_df, df_features = utilities_obj.dataset_splitting(ALL_DATASET_CSV_PATH, DATASET, Round,
    ##*                                                                              Mix_or_stable)
    utilities_obj.dataset_splitting(ALL_DATASET_CSV_PATH, DATASET, Round, Mix_or_stable)

    # exit()
    # ---------------- Process normal data ----------------

    if Mix_or_stable == '0' and DATASET == 'S_BGL':  # Stable
        # Create folder to save splits
        print(f"{GRAY}Processing normal data portion in the dataset...{RESET}")
        save_path = os.path.join(f"../datasets/{DATASET}", f"{Round}_{DATASET}_'Stable'_Splitted_Datasets")
    elif Mix_or_stable == '1' and DATASET == 'S_BGL':  # Mix
        # Create folder to save splits
        print(f"{GRAY}Processing normal data portion in the dataset...{RESET}")
        save_path = os.path.join(f"../datasets/{DATASET}", f"{Round}_{DATASET}_'Mix'_Splitted_Datasets")
    else:

        print(f"{GRAY}Processing normal data portion in the dataset...{RESET}")
        save_path = os.path.join(f"../datasets/{DATASET}", f"{Round}_{DATASET}_Splitted_Datasets")
    '''
    # first paper :*****************
    #train_df = pd.read_pickle(os.path.join(save_path, "train_df.pkl"))
    #val_df = pd.read_pickle(os.path.join(save_path, "val_df.pkl"))
    #test_df = pd.read_pickle(os.path.join(save_path, "test_df.pkl"))

    # Second paper :************
    #train_df_path = f"../../NovaAD_Plus/{DATASETS_FOLDER}/{DATASET}/{Round}_{DATASET}_Splitted_Datasets/{Round}_{DATASET}_train_df.pkl"
    #test_df_path  = f"../../NovaAD_Plus/{DATASETS_FOLDER}/{DATASET}/{Round}_{DATASET}_Splitted_Datasets/{Round}_{DATASET}_test_df.pkl"
    #val_df_path  = f"../../NovaAD_Plus/{DATASETS_FOLDER}/{DATASET}/{Round}_{DATASET}_Splitted_Datasets/{Round}_{DATASET}_val_df.pkl"


    # Third paper :************
    train_df_path = f"../../LWADLS/{DATASETS_FOLDER}/{DATASET}/{Round}_{DATASET}_Splitted_Datasets/train_df.pkl"
    test_df_path  = f"../../LWADLS/{DATASETS_FOLDER}/{DATASET}/{Round}_{DATASET}_Splitted_Datasets/test_df.pkl"
    val_df_path  = f"../../LWADLS/{DATASETS_FOLDER}/{DATASET}/{Round}_{DATASET}_Splitted_Datasets/val_df.pkl"


    '''

    train_df = pd.read_pickle(train_df_path)
    test_df = pd.read_pickle(test_df_path)
    val_df = pd.read_pickle(val_df_path)

    final_train_with_test_with_val = utilities_obj.processing_data_portion(train_df, val_df, test_df)
    # exit()
    
    
   

    # ---------------- Features Extracting ----------------
    print(f"{GRAY}Extracting features for training and test datasets...{RESET}")
    number_component, best_topic_number = features_extracting_obj.features_extracting_configuring_tuning(
        features_extracting_obj, DOC_TOPIC_DF_PATH, SENTIMENT_DF_PATH, DATASET, PRE_FINAL_GLOBAL_FEATURES_PKL_PATH,
        final_train_with_test_with_val)

    #exit()

    '''
    final_train_with_test_with_val = pd.read_pickle(PRE_FINAL_GLOBAL_FEATURES_PKL_PATH)
    final_train_with_test_with_val.info()
    dim = len(final_train_with_test_with_val['reduced_embedding'].iloc[0])
    print("Embedding dimension:", dim)
    number_component = dim
    print(number_component)
    # exit()



    # ---------------- Features Engineering: Aggregation/Transformation ----------------
    print(f"{GRAY}Aggregating and transforming features...{RESET}")

    # ------- new
    # RL_Early minimal change:
    # original full TRAIN vectors + 25/50/75% TRAIN prefixes;
    # DEV unchanged; TEST represented by sequential cumulative prefixes.
    sequences_df, X_sequences_df, y_sequences_df, m_train_normal, m_train_unlabeled, m_val, m_test = \
        features_engineering_obj.features_aggregation_transformation_early(
            final_train_with_test_with_val,
            DATASET,
            train_prefix_fractions=(0.25, 0.50, 0.75),
            test_stride=1,
        )

    x_train_normal_labelled = X_sequences_df[m_train_normal]
    x_unlabeled_from_train = X_sequences_df[m_train_unlabeled]
    ground_truth_unlabeled_data_from_train = sequences_df.loc[m_train_unlabeled, 'Label'].to_numpy()

    # ---------------- Novelty detection and label establishment ----------------
    print(f"{GRAY}Performing novelty detection and establishing labels...{RESET}")
    X_train, y_train, X_test, y_test_truth, X_val, y_val_truth = features_engineering_obj.novelty_detection_label_establishment(
        X_sequences_df, sequences_df, x_train_normal_labelled, x_unlabeled_from_train,
        ground_truth_unlabeled_data_from_train, m_train_normal, m_train_unlabeled, m_val, m_test)
    # ---------------- Anomaly Detection ----------------
    print(f"{GRAY}Running anomaly detection on test dataset...{RESET}")
    y_prefix_truth, y_prefix_pred, fit_time, predict_time = anomaly_detection_obj.anomaly_detector(
        X_train, y_train, X_test, y_test_truth, X_val, y_val_truth, mode
    )

    # ---------------- Convert prefix predictions to one decision per TEST sequence ----------------
    test_meta = sequences_df.loc[
        m_test,
        ['Original_Node_block_id', 'Prefix_End', 'Sequence_Length', 'Label', 'Is_Full']
    ].reset_index(drop=True).copy()

    test_meta['Prefix_Prediction'] = pd.Series(y_prefix_pred).astype(int).values
    X_test_prefix = list(X_test)

    y_test_truth_seq = []
    y_test_pred_seq = []
    detection_steps = []
    sequence_lengths = []
    X_test_seq = []

    for original_id, group in test_meta.groupby('Original_Node_block_id', sort=False):
        group = group.sort_values('Prefix_End')
        positions = group.index.to_numpy()

        true_label = int(group['Label'].iloc[0])
        seq_len = int(group['Sequence_Length'].iloc[0])

        alert_rows = group[group['Prefix_Prediction'] == 1]
        if len(alert_rows) > 0:
            seq_pred = 1
            first_alert = int(alert_rows['Prefix_End'].iloc[0])
        else:
            seq_pred = 0
            first_alert = seq_len

        # Use the final/full prefix vector only as the representative X_test row.
        full_pos = int(group['Prefix_End'].idxmax())

        y_test_truth_seq.append(true_label)
        y_test_pred_seq.append(seq_pred)
        detection_steps.append(first_alert)
        sequence_lengths.append(seq_len)
        X_test_seq.append(X_test_prefix[full_pos])

    y_test_truth_seq = pd.Series(y_test_truth_seq).to_numpy()
    y_test_pred_seq = pd.Series(y_test_pred_seq).to_numpy()
    detection_steps = pd.Series(detection_steps).to_numpy()
    sequence_lengths = pd.Series(sequence_lengths).to_numpy()

    # ---------------- Model Evaluation ----------------
    print(f"{GRAY}Evaluating NovaADLS-Early sequence-level performance...{RESET}")
    model_evaluation_obj.evaluation(
        Round, X_train, y_train, number_component,
        y_test_truth_seq, y_test_pred_seq, DATASET, X_test_seq,
        method_name="NovaADLS-Early",
        detection_steps=detection_steps,
        sequence_lengths=sequence_lengths,
       # training_time_min=fit_time,
       # evaluation_time_min=predict_time,
    )

    #print(f"Model training completed in {fit_time:.2f} minutes")
   # print(f"Sequential prefix prediction completed in {predict_time:.2f} minutes")


if __name__ == "__main__":
    main()
