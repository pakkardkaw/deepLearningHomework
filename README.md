# Performance Comparison between Traditional Machine Learning and Deep Learning Models on Classification Datasets
## Executive Summary: 
- Results comparing performances
- Key discussion, which one is better/ more suitable with classification on tabular datasets
- Recommendation based on dataset type

## Introduction: 
This project aims to compare the performance of various traditional Machine Learning Models using scikit-learn algorithms and Deep Learning models using TensorFlow on 5 different classification datasets including binary and multiclass classification. The classification performance was evaluated by Accuracy score and training time used for each model. 

## Assumption
Based on recent research, traditional ML models perform classification better than deep learning models on tabular datasets.

## Data
### Data source
- [Multiclass classification] Wine quality: https://www.kaggle.com/datasets/rajyellow46/wine-quality (UCI)
- [Binary classification] Credit card churn prediction: https://www.kaggle.com/datasets/anwarsan/credit-card-bank-churn
- [Binary classification] Predict Bank term deposit campaign: https://www.kaggle.com/datasets/prakharrathi25/banking-dataset-marketing-targets 
- [Binary classification] Airline customer satisfaction: https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction
- [Binary classification] Customer will claim/not claim car insurance: https://www.kaggle.com/datasets/sagnik1511/car-insurance-data

### Data Preparation:
- Cleaning data: Remove null values, replace null values with Mean or Median
- Encoding categorical data: Using get.dummie function

### EDA
Histrogram plot to check if data is imbalanced or not and observe data distribution in each class of datasets
Heatmap for features correlations

### EDAData pre-processing (Table)
- Drop uncorrelated features
- Remove extreme outliers
- Imbalanced dataset: Wine quality, Credit card churn, Car insurance
- Using random oversampling
- Data splitting (train/val/test)
- Cross validation*
  - Test 20%
  - Train 70%
  - Validate 10%
- Normalizing datasets

### Traditional Machine Learning Models:
- 3 Repetitives with fixed seed number :  round 1 = seed 911, round 2 = seed 444, round 3 = seed 888
- ML Models
  - Linear Discriminant Analysis
  - Quadratic Discriminant Analysis
  - AdaBoost
  - Bagging
  - Extra Trees Ensemble
  - Gradient Boosting
  - Random Forest
  - Ridge
  - SGD
  - BNB
  - GNB
  - KNN
  - MLP
  - LSVC
  - DTC
  - ETC
  - XGB
  - XGBRF
  - LGBM
  - SVM
- Choose the best ML results based on average accuracy score in each data sets 
- Hyperparamaters Tuning

| Datasets  |  Best ML Model   | Best Parameters  | Accuracy score  | Training time (sec)  |
|-----------|------------------|------------------|-----------------|----------------------|
|Wine quality   |   |   |   |   |
|Credit card churn prediction|   |   |   |   |
|Predict Bank term deposit campaign|   |   |   |   |
|Airline customer satisfaction|   |   |   |   |
|Customer will claim/not claim car insurance|   |   |   |   |



Train vs Validation vs Test

Datasets
Train
Validate
Test
Overfit/ Underfit/Good fit
Accuracy Score
Training time (sec)
Accuracy Score
Training time (sec)
Accuracy Score
Training time (sec)


















































































Overfit/Underfit summary

Deep Learning Model:
Network architecture: 

Drawing picture pasted from ipad




Selecting Optimizer:
- Adamax
- Nadam

รายละเอียดต่าง ๆ ของโมเดลที่เลือกใช้ (เช่น จำนวนและตำแหน่งการวาง layer, จำนวน nodes, activation function, regularization) ในรูปแบบของ network diagram หรือตาราง (โดยใส่ข้อมูลให้ละเอียดพอที่คนที่มาอ่าน จะสามารถไปสร้าง network ตามเราได้) 
o Training: รายละเอียดของการ train และ validate ข้อมูล รวมถึงทรัพยากรที่ใช้ในการ train โมเดลหนึ่ง ๆ เช่น training strategy (เช่น single loss, compound loss, two-step training, end-to-end training), loss, optimizer (learning rate, momentum, etc), batch size, 
epoch, รุ่นและจำนวน CPU หรือ GPU หรือ TPU ที่ใช้, เวลาโดยประมาณที่ใช้ train โมเดลหนึ่งตัว ฯลฯ 
Training Strategy:
Single loss?
Hyperparameters Tuning
- Start tuning with batch size (32,64,128)
- Select best batch size and tune with Epoch (100,300,500)
- Use GPU on Colab: Spec
Datasets
Deep learning Model Parameters
(Epoch (100,300,500), Batch size(32,64,128)
Optimizer (Adamax, Nadam)
Accuracy Score
Training time (sec)




















































Train vs Validation vs Test

Datasets
Train
Validate
Test
Overfit/ Underfit/Good fit
Accuracy Score
Training time (sec)
Accuracy Score
Training time (sec)
Accuracy Score
Training time (sec)


















































































Overfit/Underfit summary


Results: 
Graph comparison between best traditional ML and deep learning model → Boxplot + Scatter paring to compare ML vs Deep learning
Table showing accuracy mean±SD, parameters, time in each datasets



Datasets
Best Machine Learning Model
Best Deep Learning Model
Accuracy Score
Training time (sec)
Accuracy Score
Training time (sec)























































Accuracy score → Confusion matrix comparison between best ML and best deep learning in each dataset
Inferent run time → Inference comparison

o Discussion: 
- following assumption.
- imbalanced data → suggests further analysis


o Conclusion: 
- Summary bullets from discussion 
(Note objective to compare!!!) 


o References: 
- scikitlearn
- tensorflow
- Papers 


❑ Citing: ในกรณีที่มีคนอยาก cite (อ้างอิง) งานหรือ dataset ของเรา เราอยากให้เขา cite เราว่าอย่างไร ส่วนใหญ่นิยมเขียนในรูปแบบของ bibtex format ตามตัวอย่างในภาพ 
