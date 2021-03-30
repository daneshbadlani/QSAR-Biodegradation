# QSAR Biodegradation

- Danesh Badlani
- Sam Bluestone

## EDA

This dataset contains 41 attributes and 1 target variable with 1055 rows. We binarized the target variable, replacing "NRB" and "RB" with 0 and 1 respectively. There are some missing values in the dataset, which we imputed by using the mean strategy. We tried both strategies such as the most frequent one as well as the mean but the results turned out to be the same for both of them and mean strategy is more relevant here because the data is numerical. We also decided not to remove the missing values altogether because we would have lost around 30% of the values then. There was also the problem of class imbalance, i.e., 'NRB' values were almost double of 'RB' values. Hence, we tried both the strategies of upsampling and downsampling but ended up sticking with upsampling because we were able to get more data as well as better overall results. The correlation matrix shows highly positive correlation between SpMax_A and SpMax_L, SpMax_A and SM6_L, and SM6_L and SpMax_L while negative correlation between SdssC and Psi_i_A.

## Feature Extraction

We performed cross-validation grid search on three feature extraction methods such as Principal Component Analysis (PCA), Kernel Principal Component Analysis (KPCA) and Linear Discriminant Analysis (LDA) as well as without it to get the optimal combination of values (see results below). We used 1 component for each feature extraction methods since it has be at least 1 less than the number of target class, which is 2 here. For KPCA, we specified the kernel to be 'rbf' with gamma values of 15. We also normalized the data for LDA since it tends to assume that the data is normally distributed. LDA is also a more relevant feature extraction technique because it works better for supervised data, which is what we have here. 

## Feature Selection

For feature selection we performed two methods: Random Forest, and Sequential Backwards Search (SBS).

SBS is a greedy feature selection algorithm while random forest uses a more robust approach by measuring the impurity of each feature. See rf_selection.png and sbs.png for the results of each feature selection method.

For Random Forest, we created a RandomForest classifier with 500 subtrees. We then trained the classifier on our dataset and grabbed the feature importances for each feature, and then compiled a list of features sorted by their importance to classification according to the Random Forest model.

For SBS, we created an SVM model for our classifier with the same hyperparameters that we used for our final model (see the hyperparameter section below for the exact values) in an effort effectively determine the importance of each feature to our final model. The importance of the feaure was determined by how much each feature contributed to performance loss (in our case, we measured accuracy) if it was removed.

### Train/Test Split

We tried different sizes for the testing set between 20-30% and it did not cause a lot of variation in the results, so we stuck with a 30% size for the testing set to ensure that the model is not overfit to the training set.

### Hyperparameter Selection

We selected the appropriate hyperparameters using the GridSearchCV object from the sklearn api to perform 10-fold cross-validation. We did a cross-validation grid search on each of the feature extraction methods we used (PCA, KPCA, and LDA) as well as a grid search on the features without performing feature extraction. We tested a range of parameters for C and gamma, and tested on rbf and linear kernels. We also passed in the 'balanced' for the 'class_weight' parameter to ensure that we are testing and training on a proportional number of RBs and NRBs in the target set. Here are the results from the grid search:

```
Grid search for No Feature Extraction
Accuracy: 0.923290553334736
Parameters: {'C': 100.0, 'gamma': 0.1, 'kernel': 'rbf'}


Grid search for LDA
Accuracy: 0.8813170629076372
Parameters: {'C': 1.0, 'gamma': 1000.0, 'kernel': 'rbf'}


Grid search for PCA
Accuracy: 0.7617189143698717
Parameters: {'C': 1000.0, 'gamma': 1000.0, 'kernel': 'rbf'}


Grid search for KPCA
Accuracy: 0.8885545970965707
Parameters: {'C': 50, 'gamma': 75, 'kernel': 'rbf'}


CV Accuracy: 0.889 +/- 0.021
```

The grid search results showed pretty decisively that performing no feature extraction is the best way forward for this dataset. This is evidenced by the average accuracy (~92%) which was highest of all of the feature extraction methods as well as the hyperparameters chosen. The C value serves as a regularization parameter with higher values leading to an increased decision region and a smaller hyperplane that separates our data. The grid search for no feature extraction had a realtively high C value (100) which means that the hyperplane region is tight, but not too tight as to mischaracterize the complexity of the data. The gamma parameter tells us how much our model is influenced by each individual training example. The higher the gamma value, lower the influence is for each training example, while the lower he gamma value, the higher the influence is for each training example. The gamma value of .1 means that we are not over reacting to new datapoints, but we are also not overgeneralizing our model by putting very little weight into each new training example. Since both the C and gamma values are not extreme, we are encouraged that the evidence implies that we have an effective, generalizable model. However, to be sure, we must see how the model fairs on the testing set.

The next step was to choose which features would be included. Since we did both SBS and Random Forest feature selection, we tested our model with both methods. For each feature selection method, we compiled the list of features for the dataset in order of importance that each feature selection method determined. Then, we tracked the accuracy, precision, recall, and f1-score of the model with an increasing number of features. You can see the results for each feature selection method in RF_fs_metrics.png and SBS_fs_metrics.png. What we found was that random forest feature selection using 29 features provided the best results. The features used were: SpMax_B(m), SpMax_L, SM6_B(m), SpPosA_B(p), SpMax_A, SdssC, SM6_L, HyWi_B(m), Psi_i_A, F02[C-N], nHM, Mi, J_Dz(e), nN, F03[C-N], TI2_L, LOC, Me, n0, C%, Sd0, F04[C-N], F03[C-0], nCb-, C-026, Psi_i_1d, NssssC, nCp, and nX (listed in order of importance assigned by random forest feature selection). This achieved 93% accuracy, 96% precision, 91% recall, and a 93% F1-score.

Another important note is that not only did the accuracy of the model not siginificantly decrease from cross-validation to the testing set (which would have shown that our model was suffering frorm overfitting), but our model actually slightly improves over the accuracy of the validation set. Because of this, we can safely conclude that our final model is generalizable.

### Scalers

Since we decided to go without feature extraction, it was necessary to try several scalers. The four scalers we tried were: StandardScaler, MinMaxScaler, MaxAbsScaler, and Normalizer. They all performed relatively similarly with the StandardScaler performing slightly better than the rest. So, we stuck with the standard scaler for our final model.

### Resources

Mansouri, K., Ringsted, T., Ballabio, D., Todeschini, R., Consonni, V. (2013). Quantitative Structure - Activity Relationship models for ready biodegradability of chemicals. Journal of Chemical Information and Modeling, 53, 867-878
