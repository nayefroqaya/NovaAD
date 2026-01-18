# **NovaAD**
This is the basic implementation of our submission in ACM UMAP 2026. NovaAD: Semi-Supervised Anomaly Detection in Log Series

## 📌 Description
NovaAD is a novel approach for log-based anomaly detection, i.e., textual event series. We conducted empirical evaluations on four open-source datasets, HDFS, BGL, Thunderbird , and Spirit and achieved the following results:  
NovaAD attains the highest F1 score on the HDFS dataset and a competitive F1 score on the BGL dataset. Additionally, it demonstrates efficient runtime performance during both training and testing, achieving the shortest runtime compared to baseline methods on both GPU and CPU.

---

## Project Structure
<pre>
├─ datasets/               # Main entry point for NovaAD datasets  
├─ drain_parser/           # Configuration and parser scripts for Drain  
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
Steps to run NovaAD:

1. Install all required libraries from the requirements file (e.g., NovaAD/requirements.txt).
2. Create a dataset directory under `datasets` (e.g., `HDFS`, `BGL`,`TH_1G`, `TH_2G`, `SP_150MB`) and upload the (datasetname.log) to this directory.
3. In main.py, set the dataset name to either BGL or HDFS.
4. For Drain parser details, see [IBM Drain](https://github.com/logpai/logparser/tree/main/logparser/Drain).
5. The parsing code is available in the `drain_parser` folder.
6. Specify the dataset name in `demo.py` (e.g., BGL). The code is avaible for all datasets. Uncomment the lines of the dataset you need to use

---

## 📌 Data Parsing
1. For data parsing, all libraries are specified with their versions in the requirements file (e.g., drain_parser/requirements.txt). 
2. To start the parsing process, run (drain_parser/demo.py). 
3. The parsing output will be generated and saved in the datasets directory.

---

## 🚨 Anomaly Detection 
To apply the NovaAD pipeline on log data:
* Run the main function (`src/main.py`).
* The main function executes all stages as one pipeline: data preprocessing, anomaly detection, and evaluation.

---

