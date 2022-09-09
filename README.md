# Performance Comparison between Traditional Machine Learning and Deep Learning Models on Classification Datasets
## Executive Summary: 
- The traditional ML model performs classification better than deep learning model in both accuracy score (+6%) and training time (-641%).
- Oversampling helps improving model training in imbalanced dataset.

## Introduction: 
This project aims to compare the performance of various traditional Machine Learning Models using scikit-learn algorithms and Deep Learning models using TensorFlow on 5 different classification datasets including binary and multiclass classification. The classification performance was evaluated by Accuracy score and training time used for each model. 

## Assumption:
Based on recent research, traditional ML models perform classification better than deep learning models on tabular datasets.

## Data:
### Data source
- [Multiclass classification] Wine quality: https://www.kaggle.com/datasets/rajyellow46/wine-quality
- [Binary classification] Credit card churn prediction: https://www.kaggle.com/datasets/anwarsan/credit-card-bank-churn
- [Binary classification] Predict Bank term deposit campaign: https://www.kaggle.com/datasets/prakharrathi25/banking-dataset-marketing-targets 
- [Binary classification] Airline customer satisfaction: https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction
- [Binary classification] Customer will claim/not claim car insurance: https://www.kaggle.com/datasets/sagnik1511/car-insurance-data

## Data Preparation:
Prior to explore with the data, every dataset that has null values was treated by either removing or replacing with Mean or Median.

## Exploratory Data Analysis:
The datasets were explored and visualized.
-	Histrogram plot: To check data balancing and data distribution
-	Heatmap: To observe features correlations
-	Box plot: To observe extreme outliers

## Data Pre-processing:
Prior to train with Machine learning models, the categorical features and imbalanced dataset should be treated as below.
-	Categorical features: Encoding the features as numerical. 
-	Imbalanced dataset: Treating by using random oversampling method. 
![Encode](https://user-images.githubusercontent.com/88955224/189154290-8ae7d099-ba2c-4224-bd7a-e88f2cc9e5c2.JPG)

## Data Splitting (Train/Validate/Test):
In this study, all datasets were split into 3 sets (Test 20%, Train 70%, Validate 10%). Only train sets were normalized. Number of records in each dataset are shown below.
| Dataset | No.of Train|No.of Validate|No.of Test|
|----------|-------|-------|------|
|Wine Quality|5,856|	837	|1,674|
|Credit Card|11,900 |1,700|3,400|
|Marketing Targets|55,890|	7,985|	15,969|
|Airline Customer Satisfaction|72,732	|10,391	|20,781|
|Car Insurance|9,354 |	1,337|	2,673|


##	Traditional Machine Learning Models:
All datasets were trained 3 rounds by using traditional machine learning models (Scikit-learn) listed below;
-	Linear Discriminant Analysis
-	Quadratic Discriminant Analysis
-	AdaBoost Classifier (Adaptive Boosting)
-	Bagging Classifier
-	Extra-Trees Ensemble Classifier 
-	Gradient Boosting Classifier
-	Random Forest Classifier
-	Ridge Classifier
-	Stochastic gradient descent (SGD)
-	BernoulliNB
-	GaussianNB
-	K-Nearest neighbors Classifier
-	Multi-layer Perceptron Classifier (MLP)
-	Support Vector Machines (SVM ; SVC , LSVC)
-	Decision Tree Classifier
-	Extra Tree Classifier
-	XGBoost Classifier (eXtreme Gradient Boosting)
-	XGBoost Random Forest Classifier (XGBRF)
-	LightGBM Classifier
-	SVM (SVC)
The best machine learning model for each dataset was chosen based on average accuracy score, and then was proceeded further with Hyperparamaters Tuning. The results of the best machine learning model for each dataset are shown below.

![BestML](https://user-images.githubusercontent.com/88955224/189154474-150aedd3-8076-482d-b55a-3e0959190aaa.JPG)

-	From above table, Extra Trees Ensemble is the best machine learning model for all datasets with accuracy score ranging from 0.86 to 0.99 and training time ranging from 0.41 to 139.26 seconds.

##	ML: Train vs Validate vs Test Results
The best tuned machine learning model, Extra Trees Ensemble, was applied to Train, Validate, and Test sets. The results are shown below.

![ML_Train_Valid_Test](https://user-images.githubusercontent.com/88955224/189154552-d509722c-a361-403d-8dd0-e2a54f692a34.JPG)

-	From above table, accuracy scores of validate and test sets in each dataset are not significantly different. This indicates that the models are good fit. 

## 	Deep Learning Model:
- Creating Network architecture: 
  - In each dataset, the number of input values are equal to number of features and there is only one output value for classification. The input values are normalized before being used in the network. 
  - The baseline neural network model has 3 hidden layers, containing 32, 64, and 32 units, respectively. Non-saturating activation function (ReLU) was applied in each hidden layer. Batch normalization was placed after the activation function. Dropout is applied between the last hidden layer and the output layer with dropout rate of 30%.
  - For the output, softmax activation function is applied. 

![DL_Network](https://user-images.githubusercontent.com/88955224/189150905-8b4643ea-9374-4efc-9faa-79ab76d74fed.JPG)

- Selecting Optimizer:
After we created our network architecture, next step is to tune the parameters.
Our strategy is to use default optimizer ‘Adam’ to tune for best batch size (32,64,128) and best epoch (50,100,300), respectively. Then, the best parameters for each dataset were applied with different optimizers including Adam, Adamax, and Nadam to find the best optimizer.

- Training:
  - The training strategy is to use single loss and accuracy score to find the best model with learning rate 0.001. 
  - GPU = GPU 0: Tesla T4 (UUID: GPU-7dbe7278-372c-45fe-7bdf-0f2dac6edd94) TensorFlow 2.8.2
  - The best parameters and optimizer for each dataset are shown as below.

![DeepLpara](https://user-images.githubusercontent.com/88955224/189154748-ba29a4fe-c616-4951-80b3-e4588b8d8fcc.JPG)

## 	DL: Train vs Validate vs Test Results
The tuned deep learning model was applied to Train, Validate, and Test sets. The results are shown below.

![DL_Train_Valid_Test](https://user-images.githubusercontent.com/88955224/189154823-cf888319-7096-4562-b29a-8f4e94f86e9b.JPG)

- From above table, accuracy scores of validate and test sets in each dataset are not significantly different. This indicates that the models are good fit. 

## Results: 
-	The table below shows the comparison between best traditional ML and deep learning model for each dataset. 
![MlvsDL](https://user-images.githubusercontent.com/88955224/189154914-0741f2e0-c303-4d92-b4b5-86c81ea94e52.JPG)

-	The accuracy scores of ML model for 4 out of 5 datasets are higher than from Deep learning model. While DL model performs slightly better than ML model in Airline Customer Satisfaction.
-	The training time using in Deep learning model is significantly longer than ML model for all datasets.

![acc](https://user-images.githubusercontent.com/88955224/189160166-d950b1cd-a870-4fec-808e-983dac7cfc4d.JPG)

![tt](https://user-images.githubusercontent.com/88955224/189160188-855f2b15-79ba-4996-b31f-5a3233efb24a.JPG)

## Discussion: 
- From the comparison results, the traditional ML model (Extra Tree Assemble) performs classification better than deep learning model for 80% of the datasets based on average accuracy scores which is aligned with our assumption. 
- In terms of training time, the traditional ML model performs significantly faster for all the datasets because Machine learning uses algorithms to parse data, learn from that data, and make informed decisions based on what it has learned. While Deep learning model takes longer time to structures algorithms in layers to create an "artificial neural network” that can learn and make intelligent decisions on its own
- Oversampling method helps improving model training in imbalanced dataset. The result in table below shows the comparison of accuracy scores between oversampled and original dataset. 

![ImbalancedCP](https://user-images.githubusercontent.com/88955224/189154991-6cd68874-5dd4-4345-ab57-237087c53da1.JPG)

## Conclusion: 
- The traditional ML model performs classification better than deep learning model in both accuracy score and training time.
- Oversampling helps improving model training in imbalanced dataset.

## References:
- Scikitlearn
- Tensorflow
- Sheikh Amir Fayes et al. Is Deep Learning on Tabular Data Enough?. International Journal of Advanced Computer Science and Applications [Online] Vol.13, No.4, 2022: 466-473.
- R. Shwartz-Ziv and A. Armon. Tabular data: Deep Learning is not all you need. Information Fusion 81 [Online], 2022: 84-90. 

## Group Members:
- 6410422014 (24%): Credit card dataset & Code Review
- 6410422012 (22%): Car Insurance & Code Review
- 6410422019 (18%): Wine Quality
- 6410422023 (18%): Airline Customer Satisfaction
- 6410422035 (18%): Marketing Targets

- This project is a part of subject DADS7202. Data Analytics and Data Science. NIDA
