import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # ⛔ Disable GPU completely
import warnings
import numpy as np
import colorama
import pandas as pd
import torch

from anomaly_detection import AnomalyDetector
from features_engineering import FeaturesEngineering
from features_extracting import FeaturesExtractor
from logdata_read import LogdataRead
from model_evaluation import ModelEvaluation
from utility import Utilities
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

    #DATASET = 'SP_150MB_ratio'
    #DATASETS_FOLDER = 'datasets'
    #Round = '1'
    #mode = 'M'  # M multi classifier - S single classifier
    #Mix_or_stable = '0'  # 0 Full stable subset  / 1 mix subset

    # Paths first paper
    #ALL_DATASET_LOG_PATH = f'../{DATASETS_FOLDER}/{DATASET}/{DATASET}.LOG'

    # path for full data CSV file
    #ALL_DATASET_CSV_PATH = f'../{DATASETS_FOLDER}/{DATASET}/{DATASET}.csv' # data first paper
    #ALL_DATASET_CSV_PATH = f'../../NovaAD_Plus/{DATASETS_FOLDER}/{DATASET}/{DATASET}.csv' # data second paper

    # Path features results- paper1
    #DOC_TOPIC_DF_PATH = f'../{DATASETS_FOLDER}/{DATASET}/{Round}_{DATASET}_All_doc_topic_df.pkl'
    #SENTIMENT_DF_PATH = f'../{DATASETS_FOLDER}/{DATASET}/{Round}_{DATASET}_All_sentiment_df.pkl'
    #PRE_FINAL_GLOBAL_FEATURES_PKL_PATH = f'../{DATASETS_FOLDER}/{DATASET}/{Round}_{DATASET}_All_pre_final_global_features.pkl'

    # Path features results- paper2:
    #DOC_TOPIC_DF_PATH = f'../../NovaAD_Plus/{DATASETS_FOLDER}/{DATASET}/{Round}_{DATASET}_All_doc_topic_df.pkl'
    #SENTIMENT_DF_PATH = f'../../NovaAD_Plus/{DATASETS_FOLDER}/{DATASET}/{Round}_{DATASET}_All_sentiment_df.pkl'
    #PRE_FINAL_GLOBAL_FEATURES_PKL_PATH = f'../../NovaAD_Plus/{DATASETS_FOLDER}/{DATASET}/{Round}_{DATASET}_All_pre_final_global_features.pkl'

    # Path features results- paper3:
    #DOC_TOPIC_DF_PATH = f'../../LWADLS/{DATASETS_FOLDER}/{DATASET}/{Round}_{DATASET}_All_doc_topic_df.pkl'
    #SENTIMENT_DF_PATH = f'../../LWADLS/{DATASETS_FOLDER}/{DATASET}/{Round}_{DATASET}_All_sentiment_df.pkl'
    #PRE_FINAL_GLOBAL_FEATURES_PKL_PATH = f'../../LWADLS/{DATASETS_FOLDER}/{DATASET}/{Round}_{DATASET}_All_pre_final_global_features.pkl'


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
    #train_df_path = f"../../LWADLS/{DATASETS_FOLDER}/{DATASET}/{Round}_{DATASET}_Splitted_Datasets/{Round}_{DATASET}_train_df.pkl"
    #test_df_path  = f"../../LWADLS/{DATASETS_FOLDER}/{DATASET}/{Round}_{DATASET}_Splitted_Datasets/{Round}_{DATASET}_test_df.pkl"
    #val_df_path  = f"../../LWADLS/{DATASETS_FOLDER}/{DATASET}/{Round}_{DATASET}_Splitted_Datasets/{Round}_{DATASET}_val_df.pkl"#




    #train_df = pd.read_pickle(train_df_path)
    #test_df = pd.read_pickle(test_df_path)
    #val_df = pd.read_pickle(val_df_path)

    # ============================================================
    # Configuration
    # ============================================================

    # ============================================================
    # Cross-dataset configuration only
    # ============================================================

    DATASETS_FOLDER = "datasets"
    Round = "1"

    SOURCE_DATASETS = ["BGL"]        # example: ["BGL"], ["BGL", "TH_1G"]
    TARGET_DATASET = "HDFS"          # example: "HDFS", "SP_150MB_ratio"
    TARGET_NORMAL_FRACTION = 0.2     # fraction of normal target-train blocks
    SEED = 123

    # Cross-dataset experiment name
    # Example: BGL_To_HDFS, BGL_TH_1G_To_SP_150MB_ratio
    EXPERIMENT_NAME = "_".join(SOURCE_DATASETS) + "_To_" + TARGET_DATASET

    # Use EXPERIMENT_NAME as DATASET name for saving outputs/results
    DATASET = EXPERIMENT_NAME

    # ============================================================
    # Feature output paths - paper3 / cross dataset
    # ============================================================

    CROSS_RESULT_DIR = (
        f"../../LWADLS/{DATASETS_FOLDER}/Cross_Dataset_Results/"
        f"{EXPERIMENT_NAME}"
    )

    os.makedirs(CROSS_RESULT_DIR, exist_ok=True)

    DOC_TOPIC_DF_PATH = (
        f"{CROSS_RESULT_DIR}/"
        f"{Round}_{EXPERIMENT_NAME}_All_doc_topic_df.pkl"
    )

    SENTIMENT_DF_PATH = (
        f"{CROSS_RESULT_DIR}/"
        f"{Round}_{EXPERIMENT_NAME}_All_sentiment_df.pkl"
    )

    PRE_FINAL_GLOBAL_FEATURES_PKL_PATH = (
        f"{CROSS_RESULT_DIR}/"
        f"{Round}_{EXPERIMENT_NAME}_All_pre_final_global_features.pkl"
    )

    print(f"[INFO] Cross-dataset experiment name: {EXPERIMENT_NAME}")
    print(f"[INFO] Feature output folder: {CROSS_RESULT_DIR}")
    print(f"[INFO] DOC_TOPIC_DF_PATH: {DOC_TOPIC_DF_PATH}")
    print(f"[INFO] SENTIMENT_DF_PATH: {SENTIMENT_DF_PATH}")
    print(f"[INFO] PRE_FINAL_GLOBAL_FEATURES_PKL_PATH: {PRE_FINAL_GLOBAL_FEATURES_PKL_PATH}")

    # ============================================================
    # Dataset loader - Third paper / NovaADLS path
    # ============================================================

    def load_novaadls_dataset(DATASETS_FOLDER, DATASET_NAME, Round):
        train_df_path = (
            f"../../LWADLS/{DATASETS_FOLDER}/{DATASET_NAME}/"
            f"{Round}_{DATASET_NAME}_Splitted_Datasets/"
            f"train_df.pkl"
        )

        test_df_path = (
            f"../../LWADLS/{DATASETS_FOLDER}/{DATASET_NAME}/"
            f"{Round}_{DATASET_NAME}_Splitted_Datasets/"
            f"test_df.pkl"
        )

        val_df_path = (
            f"../../LWADLS/{DATASETS_FOLDER}/{DATASET_NAME}/"
            f"{Round}_{DATASET_NAME}_Splitted_Datasets/"
            f"val_df.pkl"
        )

        print(f"[INFO] Reading train: {train_df_path}")
        print(f"[INFO] Reading val  : {val_df_path}")
        print(f"[INFO] Reading test : {test_df_path}")

        train_df = pd.read_pickle(train_df_path)
        test_df = pd.read_pickle(test_df_path)
        val_df = pd.read_pickle(val_df_path)

        return train_df, val_df, test_df

    # ============================================================
    # Label normalization
    # ============================================================

    def normalize_label_column(df):
        """
        If the PKL has Original_Label, use it as Label.
        This keeps the rest of the pipeline unchanged.
        """
        df = df.copy()

        if "Original_Label" in df.columns:
            if "Label" in df.columns:
                df = df.drop(columns=["Label"])
            df = df.rename(columns={"Original_Label": "Label"})

        return df

    # ============================================================
    # Select fraction of normal target training blocks
    # ============================================================

    def select_target_normal_fraction(target_train_df, fraction, seed=123):
        """
        Select fraction of normal target TRAINING sequences.
        One sequence = one Node_block_id.
        """
        if fraction < 0 or fraction > 1:
            raise ValueError("TARGET_NORMAL_FRACTION must be between 0 and 1.")

        unique_normal_blocks = target_train_df[
            target_train_df["Label"] == "Normal"
        ]["Node_block_id"].unique()

        print(f"[INFO] Target normal blocks available: {len(unique_normal_blocks)}")

        if len(unique_normal_blocks) == 0 or fraction == 0:
            print("[INFO] No target normal blocks selected.")
            return target_train_df.iloc[0:0].copy()

        sample_size = int(round(len(unique_normal_blocks) * fraction))
        sample_size = max(1, sample_size)
        sample_size = min(sample_size, len(unique_normal_blocks))

        rng = np.random.default_rng(seed)

        selected_blocks = rng.choice(
            unique_normal_blocks,
            size=sample_size,
            replace=False
        )

        target_normal_fraction_df = target_train_df[
            target_train_df["Node_block_id"].isin(selected_blocks)
        ].copy()

        print(f"[INFO] Target normal fraction: {fraction}")
        print(f"[INFO] Selected target normal blocks: {sample_size}")
        print(f"[INFO] Selected target normal rows: {len(target_normal_fraction_df)}")

        return target_normal_fraction_df

    # ============================================================
    # Build cross-dataset train_df, val_df, test_df
    # ============================================================

    print("[INFO] Running cross-dataset experiment")
    print(f"[INFO] Source datasets: {SOURCE_DATASETS}")
    print(f"[INFO] Target dataset: {TARGET_DATASET}")
    print(f"[INFO] Target normal fraction: {TARGET_NORMAL_FRACTION}")

    # ------------------------------------------------------------
    # Load source train datasets
    # ------------------------------------------------------------

    source_train_list = []

    for src_dataset in SOURCE_DATASETS:
        src_train_df, _, _ = load_novaadls_dataset(
            DATASETS_FOLDER,
            src_dataset,
            Round
        )

        src_train_df = normalize_label_column(src_train_df)
        src_train_df = src_train_df.copy()

        # Prevent Node_block_id collision between datasets.
        # This does not delete rows. It only makes block IDs unique.
        src_train_df["Node_block_id"] = (
            src_dataset
            + "__train__"
            + src_train_df["Node_block_id"].astype(str)
        )

        source_train_list.append(src_train_df)

        print(
            f"[INFO] Source {src_dataset}: "
            f"{len(src_train_df)} rows, "
            f"{src_train_df['Node_block_id'].nunique()} blocks"
        )

    source_train_df = pd.concat(source_train_list, ignore_index=True)

    # ------------------------------------------------------------
    # Load target train, validation, and test
    # ------------------------------------------------------------

    target_train_df, val_df, test_df = load_novaadls_dataset(
        DATASETS_FOLDER,
        TARGET_DATASET,
        Round
    )

    target_train_df = normalize_label_column(target_train_df)
    val_df = normalize_label_column(val_df)
    test_df = normalize_label_column(test_df)

    # ------------------------------------------------------------
    # Select fraction of normal target-train blocks
    # ------------------------------------------------------------

    target_normal_fraction_df = select_target_normal_fraction(
        target_train_df,
        TARGET_NORMAL_FRACTION,
        seed=SEED
    )

    target_normal_fraction_df = target_normal_fraction_df.copy()

    # Prevent Node_block_id collision between target-normal fraction and sources.
    target_normal_fraction_df["Node_block_id"] = (
        TARGET_DATASET
        + "__target_train_normal__"
        + target_normal_fraction_df["Node_block_id"].astype(str)
    )

    # ------------------------------------------------------------
    # Final cross-dataset split
    # ------------------------------------------------------------

    train_df = pd.concat(
        [source_train_df, target_normal_fraction_df],
        ignore_index=True
    )

    print("[INFO] Cross-dataset split created")
    print(f"[INFO] Train rows : {len(train_df)}")
    print(f"[INFO] Val rows   : {len(val_df)}")
    print(f"[INFO] Test rows  : {len(test_df)}")
    print(f"[INFO] Train blocks: {train_df['Node_block_id'].nunique()}")
    print(f"[INFO] Val blocks  : {val_df['Node_block_id'].nunique()}")
    print(f"[INFO] Test blocks : {test_df['Node_block_id'].nunique()}")

    # ============================================================
    # Keep your original function unchanged
    # ============================================================

    final_train_with_test_with_val = utilities_obj.processing_data_portion(
        train_df,
        val_df,
        test_df
    )


    #final_train_with_test_with_val = utilities_obj.processing_data_portion(train_df, val_df, test_df)
    # exit()
    
    
   

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

    #exit()


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
    sequences_df, X_sequences_df, y_sequences_df, m_train_normal, m_train_unlabeled, m_val, m_test = features_engineering_obj.features_aggregation_transformation(
        final_train_with_test_with_val, DATASET)

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
    y_test_truth, y_test_pred, fit_time, predict_time = anomaly_detection_obj.anomaly_detector(X_train, y_train, X_test,
                                                                                               y_test_truth, X_val,
                                                                                               y_val_truth, mode)

    # ---------------- Model Evaluation ----------------
    print(f"{GRAY}Evaluating model performance...{RESET}")
    model_evaluation_obj.evaluation(Round, X_train, y_train, number_component, y_test_truth, y_test_pred, DATASET,
                                    X_test)

    print(f"Model training completed in {fit_time:.2f} minutes")
    print(f"Prediction completed in {predict_time:.2f} minutes")


if __name__ == "__main__":
    main()
