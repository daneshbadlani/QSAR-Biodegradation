# CSCI297-Test3

- Danesh Badlani
- Sam Bluestone

## EDA

## Feature Extraction

## Feature Selection

## Train/Test Split, Scalers, and Hyperparameter Selection

### Train/Test Split

We tried different sizes for the testing set between 20-30% and it did not cause a lot of variation in the results, so we stuck with a 30% size for the testing set to ensure that the model is not overfit to the training set.

### Hyperparameter Selection

We selected the appropriate hyperparameters using the GridSearchCV object from the sklearn api to perform 10-fold cross-validation. We did a cross-validation grid search on each of the feature extraction methods we used (PCA, KPCA, and LDA) as well as a grid search on the features without performing feature extraction. We tested a range of parameters for C and gamma, and tested on rbf and linear kernels. Here are the results:

```
Grid search for No Feature Extraction
Accuracy: 0.8781007034431692
Parameters: {'C': 100.0, 'gamma': 0.01, 'kernel': 'rbf'}


Grid search for LDA
Accuracy: 0.8740096260644206
Parameters: {'C': 10.0, 'gamma': 0.1, 'kernel': 'rbf'}


Grid search for PCA
Accuracy: 0.7534801925212884
Parameters: {'C': 1.0, 'gamma': 0.1, 'kernel': 'rbf'}


Grid search for KPCA
Accuracy: 0.6626064420584968
Parameters: {'C': 0.0001, 'kernel': 'linear'}
```

Although the results show that the highest accuracy was achieved without feature extraction (but not by much), the parameters returned suggest that the model would be overfit. A C value of 100 is very high and a gamma value of 0.01 is very low, which would imply that our model would be overfit to the training set. Since LDA had almost the same accuracy and more reasonable values for C and gamma, we chose to stick with LDA feature extraction for our final model.

The next step was to choose which features would be included. Since we did both SBS and Random Forest feature selection, we tested our model with both methods. For each feature selection method, we compiled the list of features for the dataset in order of importance that each feature selection method determined. Then, we tracked the accuracy, precision, recall, and f1-score of the model with an increasing number of features. You can see the results for each feature selection method in RF_fs_metrics.png and SBS_fs_metrics.png. What we found was that random forest feature selection using 11 features provided the best results without using >35 of the 41 features. The features used were: SpMax_L, J_Dz(e), nHM, F01[N-N], F04[C-N], NssssC, nCb-, C%, nCp, n0, and F03[C-N] (listed in order of importance assigned by random forest feature selection). This achieved 83% accuracy, 70% precision, 77% recall, and a 74% F1-score.

### Scalers

Since we ended up using LDA for feature extraction and LDA assumes normality of the data, we using a Normalizer for the scaler.
