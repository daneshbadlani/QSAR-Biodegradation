import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

import sys

df = pd.read_csv("NewBioDeg.csv", header=None)

df.columns = [ 
    "SpMax_L",
    "J_Dz(e)",
    "nHM",
    "F01[N-N]",
    "F04[C-N]",
    "NssssC",
    "nCb-",
    "C%",
    "nCp",
    "n0",
    "F03[C-N]",
    "SdssC",
    "HyWi_B(m)",
    "LOC",
    "SM6_L",
    "F03[C-0]",
    "Me",
    "Mi",
    "nN-N",
    "nArN02",
    "nCRX3",
    "SpPosA_B(p)",
    "nCIR",
    "B01[C-Br]",
    "B03[C-Cl]",
    "N-073",
    "SpMax_A",
    "Psi_i_1d",
    "B04[C-Br]",
    "Sd0",
    "TI2_L",
    "NCrt",
    "C-026",
    "F02[C-N]",
    "nHDon",
    "SpMax_B(m)",
    "Psi_i_A",
    "nN",
    "SM6_B(m)",
    "nArC00R",
    "nX",
    "ready"

] 

feat_labels = list(df.columns)

#perform imputation to fill in missing data
df['ready'] = df['ready'].replace(to_replace={"NRB":0, "RB":1})
imr_frequent = SimpleImputer(missing_values=np.nan)
imr_frequent = imr_frequent.fit(df[[i for i in list(df.columns) if i != "ready"]].values)
df[[i for i in list(df.columns) if i != "ready"]] = pd.DataFrame(data=imr_frequent.transform(df[[i for i in list(df.columns) if i != "ready"]].values))

X = df[[i for i in list(df.columns) if i != "ready"]]
y = df['ready']

#list of features in order of importance determined by SBS feature selection
sbs_features = ['J_Dz(e)', 'nHM', 'F01[N-N]', 'F04[C-N]', 'NssssC', 'nCb-', 'C%', 'nCp',
       'n0', 'SdssC', 'HyWi_B(m)', 'LOC', 'Me', 'nN-N', 'nArN02', 'nCRX3',
       'SpPosA_B(p)', 'nCIR', 'N-073', 'SpMax_A', 'Psi_i_1d', 'B04[C-Br]',
       'NCrt', 'C-026', 'F02[C-N]', 'nHDon', 'Psi_i_A', 'nN', 'SM6_B(m)',
       'nArC00R', 'nX']


#list of features in order of importance determined by RF feature selection
rf_features = ['SpMax_L', 'J_Dz(e)', 'nHM', 'F01[N-N]', 'F04[C-N]', 'NssssC', 'nCb-',
       'C%', 'nCp', 'n0', 'F03[C-N]', 'SdssC', 'HyWi_B(m)', 'LOC', 'SM6_L',
       'F03[C-0]', 'Me', 'Mi', 'nN-N', 'nArN02', 'nCRX3', 'SpPosA_B(p)',
       'nCIR', 'B01[C-Br]', 'B03[C-Cl]', 'N-073', 'SpMax_A', 'Psi_i_1d',
       'B04[C-Br]', 'Sd0', 'TI2_L', 'NCrt', 'C-026', 'F02[C-N]', 'nHDon',
       'SpMax_B(m)', 'Psi_i_A', 'nN', 'SM6_B(m)', 'nArC00R', 'nX']


#split into testing and training set
X_train, X_test, y_train, y_test =    train_test_split(X, y, 
                     test_size=0.3, 
                     random_state=0, 
                     stratify=y)

#standardizing the dataset                     
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)



#Feature Extraction

#PCA
scikit_pca = PCA(n_components=1)
X_train_pca = scikit_pca.fit_transform(X_train_std)
X_test_pca = scikit_pca.fit_transform(X_test_std)


#KPCA
scikit_kpca = KernelPCA(n_components=1, kernel='rbf', gamma=15)
X_train_kpca = scikit_kpca.fit_transform(X_train_std)
X_test_kpca = scikit_kpca.fit_transform(X_test_std)


#LDA

#normalize the dataset before performing LDA                    
norm = Normalizer()
X_train_norm = norm.fit_transform(X_train)
X_test_norm = norm.transform(X_test)

lda = LDA(n_components=1)
X_train_lda = lda.fit_transform(X_train_norm, y_train)
X_test_lda = lda.transform(X_test_norm)



#Grid search to determine which hyper parameters are best

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 25, 50, 75, 100.0, 1000.0]

param_grid = [{'C':param_range,
               'kernel':['linear']},
              {'C':param_range,
               'gamma':param_range,
               'kernel':['rbf']}]


gs = GridSearchCV(estimator=SVC(), 
                  param_grid=param_grid,
                  scoring='accuracy',
                  refit=True,
                  cv=10,
                  n_jobs=-1)


#grid search for each feature extraction method
for X_train_gs, name in zip([X_train_std, X_train_lda, X_train_pca, X_train_kpca], ["No Feature Extraction", "LDA", "PCA", "KPCA"]):
    print("\n\nGrid search for", name)
    gs = gs.fit(X_train_gs, y_train)
    print("Accuracy:", gs.best_score_)
    print("Parameters:", gs.best_params_)
    
gs = GridSearchCV(estimator=SVC(),
                param_grid=param_grid,
                scoring='accuracy',
                cv=2)

scores = cross_val_score(gs, X_train, y_train,
                        scoring='accuracy', cv=5)

print("\n\nCV Accuracy: %.3f +/- %.3f" % (np.mean(scores), np.std(scores)))



#Hyper parameters determined from previous grid search
svm = SVC(C=10, kernel='rbf', gamma=0.1)

#Calculate accuracy, precision, recall, and f1-score using each RF and SBS feature selection
num_features = list(range(3, 41))
for name, features in zip(["SBS", "RF"], [sbs_features, rf_features]):
    print("\n\n" + name + " Feature Selection")
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []

    for num in num_features:
        #Code from https://stackoverflow.com/questions/40636514/selecting-pandas-dataframe-column-by-list
        X_train_new = X_train[X_train.columns.intersection(features[:num])]
        X_test_new = X_test[X_test.columns.intersection(features[:num])]
        
        X_train_norm = norm.fit_transform(X_train_new)
        X_test_norm = norm.transform(X_test_new)

        lda = LDA(n_components=1)
        X_train_lda = lda.fit_transform(X_train_norm, y_train)
        X_test_lda = lda.transform(X_test_norm)

        print("\n\nNumber of features:", num)
            
        svm.fit(X_train_lda, y_train)
        y_pred = svm.predict(X_test_lda)
        
        print("Accuracy:", accuracy_score(y_pred, y_test))
        print("Precision:", precision_score(y_pred, y_test))
        print("Recall:", recall_score(y_pred, y_test))
        print("F1-Score:", f1_score(y_pred, y_test))

        accuracy_list.append(accuracy_score(y_pred, y_test))
        precision_list.append(precision_score(y_pred, y_test))
        recall_list.append(recall_score(y_pred, y_test))
        f1_list.append(f1_score(y_pred, y_test))

    #plot metrics for each feature selection method
    plt.plot(num_features, accuracy_list, label='accuracy')
    plt.plot(num_features, precision_list, label='precision')
    plt.plot(num_features, recall_list, label='recall')
    plt.plot(num_features, f1_list, label='f1-score')

    plt.title(name)
    plt.xlabel('Number of Features Used')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig(name+"_fs_metrics.png")
    plt.show()





    














