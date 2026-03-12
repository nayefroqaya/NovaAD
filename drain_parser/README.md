# Drain

Drain is an online log parser that can parse logs into structured events in a streaming and timely manner. It employs a parse tree with fixed depth to guide the log group search process, which effectively avoids constructing a very deep and unbalanced tree. 

### Running

The code has been tested in the following enviornment:
+ python 3.7.6
+ regex 2022.3.2
+ pandas 1.0.1
+ numpy 1.18.1
+ scipy 1.4.1

Run the following scripts to start the demo:

```
python demo.py
```

Run the following scripts to execute the benchmark:

```
python benchmark.py
```

### Benchmark

Running the benchmark script on Loghub_2k datasets, you could obtain the following results.

|   Dataset   | F1_measure | Accuracy |
|:-----------:|:----------|:--------|
|     HDFS    | 0.999984   | 0.9975   |
|    Hadoop   | 0.998989   | 0.9475   |
|    Spark    | 0.991876   | 0.92     |
|  Zookeeper  | 0.999536   | 0.9665   |
|     BGL     | 0.999599   | 0.9625   |
|     HPC     | 0.990644   | 0.887    |
| Thunderbird | 0.999329   | 0.955    |
|   Windows   | 0.999994   | 0.997    |
|    Linux    | 0.992437   | 0.69     |
|   Android   | 0.995913   | 0.911    |
|  HealthApp  | 0.918381   | 0.78     |
|    Apache   | 1          | 1        |
|  Proxifier  | 0.784886   | 0.5265   |
|   OpenSSH   | 0.999221   | 0.7875   |
|  OpenStack  | 0.992536   | 0.7325   |
|     Mac     | 0.975451   | 0.7865   |

