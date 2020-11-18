import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix
import numpy as np
import seaborn as sns
from mlxtend.plotting import heatmap
from sklearn.ensemble import RandomForestClassifier
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
# df = df.fillna(df.mean())
imr_frequent = SimpleImputer(missing_values=np.nan)
imr_frequent = imr_frequent.fit(df[[i for i in list(df.columns) if i != "ready"]].values)
df[[i for i in list(df.columns) if i != "ready"]] = pd.DataFrame(data=imr_frequent.transform(df[[i for i in list(df.columns) if i != "ready"]].values))




"""
Random Forest Feature Selection
"""

X = df[[i for i in list(df.columns) if i != "ready"]]
y = df['ready']

feat_labels = X.columns

stdsc = StandardScaler()
X = stdsc.fit_transform(X)

forest = RandomForestClassifier(n_estimators=500,
                                random_state=1)

forest.fit(X, y)
importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]

for f in range(X.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feat_labels[indices[f]], 
                            importances[indices[f]]))

plt.title('Feature Importance')
plt.bar(range(X.shape[1]), 
        importances[indices],
        align='center')

plt.xticks(range(X.shape[1]), 
           feat_labels[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.tight_layout()
plt.savefig("rf_selection.png")
plt.show()

sfm = SelectFromModel(forest, prefit=True, threshold=.03)
X_selected = sfm.transform(X)
print('Number of features that meet this threshold criterion:', 
      X_selected.shape[1])
#print("Threshold %f" % np.mean(importances))


# Now, let's print the  features that met the threshold criterion for feature selection that we set earlier (note that this code snippet does not appear in the actual book but was added to this notebook later for illustrative purposes):
cols = []
for f in range(X_selected.shape[1]):
    cols.append(feat_labels[indices[f]])    
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feat_labels[indices[f]], 
                            importances[indices[f]]))


#Correlation heatmap
cols.append("ready")
cm = np.corrcoef(df[cols].values.T)
hm = heatmap(cm, row_names=cols, column_names=cols)
plt.show()



