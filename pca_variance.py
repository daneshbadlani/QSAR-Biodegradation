import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mlxtend.plotting import heatmap
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
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

#perform imputation to fill in missing data
df['ready'] = df['ready'].replace(to_replace={"NRB":0, "RB":1})
imr_frequent = SimpleImputer(missing_values=np.nan)
imr_frequent = imr_frequent.fit(df[[i for i in list(df.columns) if i != "ready"]].values)
df[[i for i in list(df.columns) if i != "ready"]] = pd.DataFrame(data=imr_frequent.transform(df[[i for i in list(df.columns) if i != "ready"]].values))

# df.info()

X = df[[i for i in list(df.columns) if i != "ready"]]
y = df['ready']

feat_labels = X.columns

# Splitting data into training and testing 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                     stratify=y,
                     random_state=0)

# standardizing the dataset                     
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

# plot explained variances
plt.bar(range(1,42), var_exp, alpha=0.5,
        align='center', label='individual explained variance')
plt.step(range(1,42), cum_var_exp, where='mid',
         label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.show()