# Early-Signals-in-Noisy-Network
Early Signals in Noisy Network: A Meta Hybrid Deep learning framework for Reconnaissance Detection in IoT Environments

🎯 Research Objective
The objective is to investigate whether combining optimized recurrent deep learning with gradient-boosted meta-classification can improve reconnaissance/attack detection performance in noisy and diverse IoT traffic.


🎯 Abstract:
As the IoT keeps expanding into homes, hospitals factories and urban infrastructure, cyber-attacks have started to grow in size right alongside it, even slightly out of control.  Of particular concern here are reconnaissance attacks. They do not usually inflict immediate harm, but they survey networks and scout vulnerabilities and prepare the soil before more destructive intrusions. The thing is that IoT environments are inherently nasty, Devices varies in capability, purpose and communications protocol wildly, and this makes early-stage attack detection little more straightforward than hunting down. It is against this backdrop that this work discusses a deep learning framework that is meant to learn on cic-IoT 2023 data, a dataset that mirrors real and diverse traffic scenarios instead of excessively clean lab environments. Processing the data before training can remove redundancy and isolate features, which in fact are important, at the cost of possibly useful noise being lost during this step. A number of models are then discussed such as LSTM, Bi-LSTM and meta-hybrid model approach that incorporate their merits. The findings are quite persuasive. Although the individual LSTM and Bi-LSTM models show good performance with accuracies of 90.42% and 92.45% respectively, the meta-hybrid model goes a step further and records an accuracy of 95.42%. Naturally, accuracy is not the only part of the story, and precision, recall, and F1 score are also thought to be included to create a more comprehensive picture. Combined, the results indicate that the presented framework can identify reconnaissance activity consistently and, consequently, enhance the security of IoT. Nevertheless, concerns are left regarding how effectively such a model can respond to completely new devices or invisible traffic patterns, which seem to be a logical trend of future work instead of an infirmity in itself

🧪 Methodology
1. Data Preprocessing

The CIC-IoT2023 dataset is processed through:

Duplicate removal
Null/missing-value handling
Feature normalization using Min-Max or Z-score scaling
Categorical label encoding
Feature selection/dimensionality reduction where applicable
Reshaping for LSTM-based sequence learning

🎯 The processed dataset is represented as:
D = {(xᵢ, yᵢ)}ᵢ₌₁ᴺ

🎯 The processed dataset is divided into:
Training data = 80%
Testing data  = 20%

🎯 AVOA-Optimized LSTM

LSTM is used to learn temporal/sequential representations from network traffic.

AVOA is used to optimize important LSTM parameters, including:
Number of neurons
Learning rate
Weight initialization
Dropout rate


📊 Experimental Results

| Model             | Epochs | Batch Size | Accuracy |   F1 Score | Precision | Recall |
| ----------------- | -----: | ---------: | -------: | ---------: | --------: | -----: |
| LSTM + AVOA       |     70 |         32 |   0.8800 |     0.8889 |    0.9143 | 0.8649 |
| LSTM + AVOA       |    128 |         64 |   0.9000 |     0.9083 |    0.9252 | 0.8919 |
| Meta-Hybrid Model |    200 |         64 |   0.9542 | **0.9900** |    0.9313 | 0.8744 |

Dataset used : CIC IoT dataset 2023  
link : https://www.unb.ca/cic/datasets/iotdataset-2023.html
