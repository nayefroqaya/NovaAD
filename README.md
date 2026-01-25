# **NovaADLS**
This is the basic implementation of our submission in ACM UMAP 2026. NovaADLS: Semi-Supervised Anomaly Detection in Log Series

## 📌 Description

In this paper, we propose NovaADLS, an end-to-end semi-supervised anomaly detection system for textual event series that requires only a small set of normal logs to be trained. NovaADLS eliminates manual configuration with its universal feature representation approach and automatic self-configuration mechanisms. To address log instability and semantic variations in the event stream, the system integrates semantic embedding features and uses a hierarchical ensemble of voting and stacking classifiers.
Extensive experiments on real-world datasets, including BGL, HDFS, Thunderbird, and Spirit, demonstrate that NovaADLS consistently outperforms state-of-the-art baselines in F1 score, precision, and recall; it also  offers fast training and inference times on both CPU and GPU platforms.

---

## Project Structure
<pre>
├─ datasets/               # Main entry point for NovaADLS datasets  
├─ drain_parser/           # Configuration and parser scripts for Drain  with its references
├─ src/  
│  ├─ main.py              # Main script to trigger the full pipeline  
│  ├─ logdata_read.py      # Reads parsed data and performs cleaning and column selection  
│  ├─ features_extracting.py  # Extracts features from log events  
│  ├─ features_engineering.py # Performs feature transformation  
│  ├─ anomaly_detection.py    # Contains anomaly detection modules  
│  ├─ model_evaluation.py     # Evaluation module (Precision, Recall, F1-score, etc.)  
│  └─ utility.py              # Helper functions used across different stages  
</pre>   
---

## 📊 Datasets
We used two open-source log datasets (more will be added in the future):

| Software System     | Description                        | Data Size| Link                                         |
|--------------------|------------------------------------|-----------|----------------------------------------|
| HDFS               | Hadoop Distributed File System log | 1.47 GB   | [LogHub](https://github.com/logpai/loghub)   |
| BGL                | Blue Gene/L supercomputer log      | 708.76 MB | [LogHub](https://github.com/logpai/loghub)   |
| Thunderbird (1G)   | Thunderbird supercomputer log      | 1 GB      | [LogHub](https://github.com/logpai/loghub)   |
| Thunderbird (2G)   | Thunderbird supercomputer log      | 2 GB      | [LogHub](https://github.com/logpai/loghub)   |
| Spirit (SP_150MB)  | Supercomputing system log          | 150 MB    | [Figshare](https://figshare.com/s/6d3c6a83f4828d17be79?file=27775929) |


---

## ⚙️ Environment
All libraries are specified with their versions in the requirements file (e.g., NovaAD/requirements.txt).

---

## 🛠️ Preparation
Steps to run NovaADLS:

1. Install all required libraries from the requirements file (e.g., NovaAD/requirements.txt).
2. Create a dataset directory under `datasets` (e.g., `HDFS`, `BGL`,`TH_1G`, `TH_2G`, `SP_150MB`) and upload the (datasetname.log) to this directory.
3. In main.py, set the dataset name (e.g., `HDFS`, `BGL`,`TH_1G`, `TH_2G`, `SP_150MB`)
4. For Drain parser details, see [IBM Drain](https://github.com/logpai/logparser/tree/main/logparser/Drain).
5. The parsing code is available in the `drain_parser` folder.
6. Specify the dataset name in `demo.py` (e.g., BGL). The code is available for all datasets. Uncomment the lines of the dataset you need to use

---

## 📌 Data Parsing
1. For data parsing, all libraries are specified with their versions in the requirements file (e.g., drain_parser/requirements.txt). 
2. To start the parsing process, run (drain_parser/demo.py). 
3. The parsing output will be generated and saved in the datasets' directory.

---

## 🚨 Anomaly Detection 
To apply the NovaADLS pipeline on log data:
* Before start running, you must specify the following parameters :

    DATASET = 'BGL'  # (e.g., `HDFS`, `BGL`,`TH_1G`, `TH_2G`, `SP_150MB`) .
    DATASETS_FOLDER = 'datasets'.
    Round= '1'   # we run the system three time. Round flag helps to save the data with three versions.
    Mode='M'  # M multi classifier - S single classifier # This flag is important for evaluation stage. Default is M.
    Mix_or_stable='0'  # 0 Full stable subset  / 1 mix subset  #  This flag is important for evaluation stage.

* Decide you will run on CPU or GPU
* Run the main function (`src/main.py`).
* The main function executes all stages as one pipeline: data preprocessing, anomaly detection, and evaluation.

---
