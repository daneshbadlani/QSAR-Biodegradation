# CSCI297-Test3

- Danesh Badlani
- Sam Bluestone

## EDA

This dataset contains 41 attributes and 1 target variable with 1055 rows. We binarized the target variable, replacing "NRB" and "RB" with 0 and 1 respectively. There are some missing values in the dataset, which we imputed by using the mean strategy. We tried both strategies such as the most frequent one as well as the mean but the results turned out to be the same for both of them and mean strategy is more relevant here because the data is numerical. We also decided not to remove the missing values altogether because we would have lost around 30% of the values then. There was also the problem of class imbalance, i.e., 'NRB' values were almost double of 'RB' values. Hence, we tried both the strategies of upsampling and downsampling but ended up sticking with upsampling because we were able to get more data as well as better overall results. The correlation matrix shows highly positive correlation between SpMax_A and SpMax_L, SpMax_A and SM6_L, and SM6_L and SpMax_L while negative correlation between SdssC and Psi_i_A.  

## Feature Extraction

## Feature Selection

For feature selection we performed two methods: Random Forest, and Sequential Backwards Search (SBS).

SBS is a greedy feature selection algorithm while random forest uses a more robust approach by measuring the impurity of each feature. See rf_selection.png and sbs.png for the results of each feature selection method.

## Train/Test Split, Scalers, and Hyperparameter Selection

### Train/Test Split

We tried different sizes for the testing set between 20-30% and it did not cause a lot of variation in the results, so we stuck with a 30% size for the testing set to ensure that the model is not overfit to the training set.

### Hyperparameter Selection

We selected the appropriate hyperparameters using the GridSearchCV object from the sklearn api to perform 10-fold cross-validation. We did a cross-validation grid search on each of the feature extraction methods we used (PCA, KPCA, and LDA) as well as a grid search on the features without performing feature extraction. We tested a range of parameters for C and gamma, and tested on rbf and linear kernels. Here are the results:

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

Although the results show that the highest accuracy was achieved without feature extraction (but not by much), the parameters returned suggest that the model would be overfit. A C value of 100 is very high and a gamma value of 0.01 is very low, which would imply that our model would be overfit to the training set. Since LDA had almost the same accuracy and more reasonable values for C and gamma, we chose to stick with LDA feature extraction for our final model.

The next step was to choose which features would be included. Since we did both SBS and Random Forest feature selection, we tested our model with both methods. For each feature selection method, we compiled the list of features for the dataset in order of importance that each feature selection method determined. Then, we tracked the accuracy, precision, recall, and f1-score of the model with an increasing number of features. You can see the results for each feature selection method in RF_fs_metrics.png and SBS_fs_metrics.png. What we found was that random forest feature selection using 11 features provided the best results without using >35 of the 41 features. The features used were: SpMax_L, J_Dz(e), nHM, F01[N-N], F04[C-N], NssssC, nCb-, C%, nCp, n0, and F03[C-N] (listed in order of importance assigned by random forest feature selection). This achieved 83% accuracy, 70% precision, 77% recall, and a 74% F1-score.

### Scalers

Since we ended up using LDA for feature extraction and LDA assumes normality of the data, we using a Normalizer for the scaler.
