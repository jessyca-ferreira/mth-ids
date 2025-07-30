# MTH-IDS: A Multi-Tiered Hybrid Intrusion Detection System for Internet of Vehicles

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from xgboost import plot_importance
import subprocess
import zipfile
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_selection import mutual_info_classif
from FCBF_module import FCBF, FCBFK, FCBFiP, get_i
from imblearn.over_sampling import SMOTE
import logging
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.decomposition import KernelPCA
import zipfile
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from sklearn import metrics

# Configure logging to log to a file
logging.basicConfig(
    filename='mth_ids.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logging.info("MTH-IDS script started.")
logging.info("Downloading dataset...")

subprocess.run(['wget', 'http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/CSVs/MachineLearningCSV.zip', '-O', 'CIC_IDS_2017.zip'], check=True) 

with zipfile.ZipFile("CIC_IDS_2017.zip", 'r') as zip_ref:
    zip_ref.extractall("./data")
    
df_list = []
for file in os.listdir('./data/MachineLearningCVE/'):
  df_aux = pd.read_csv(f'./data/MachineLearningCVE/{file}')
  df_list.append(df_aux)
  
df = pd.concat(df_list, ignore_index=True)

del df_list, df_aux
    
# Z-score normalization
features = df.dtypes[df.dtypes != 'object'].index
df[features] = df[features].apply(lambda x: (x - x.mean()) / (x.std()))

# Fill empty values by 0
df = df.fillna(0)

# Data sampling
labelencoder = LabelEncoder()
df.iloc[:, -1] = labelencoder.fit_transform(df.iloc[:, -1])

df_minor = df[(df['Label']==6)|(df['Label']==1)|(df['Label']==4)]
df_major = df.drop(df_minor.index)

X = df_major.drop(['Label'], axis=1)
y = df_major.iloc[:, -1].values.reshape(-1, 1)
y = np.ravel(y)

kmeans = MiniBatchKMeans(n_clusters=1000, random_state=0).fit(X)
klabel = kmeans.labels_
df_major['klabel'] = klabel

cols = list(df_major)
cols.insert(78, cols.pop(cols.index('Label')))
df_major = df_major.loc[:, cols]

def typicalSampling(group):
    name = group.name
    frac = 0.008
    return group.sample(frac=frac)

result = df_major.groupby('klabel', group_keys=False).apply(typicalSampling)

result = result.drop(['klabel'], axis=1)
result = result.append(df_minor)

result.to_csv('./data/CICIDS2017_sample_km.csv', index=0)

# Split train set and test set
df = pd.read_csv('./data/CICIDS2017_sample_km.csv')

X = df.drop(['Label'], axis=1).values
y = df.iloc[:, -1].values.reshape(-1, 1)
y = np.ravel(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0, stratify=y)

# Feature engineering
importances = mutual_info_classif(X_train, y_train)

# calculate the sum of importance scores
f_list = sorted(zip(map(lambda x: round(x, 4), importances), features), reverse=True)
Sum = 0
fs = []
for i in range(0, len(f_list)):
    Sum = Sum + f_list[i][0]
    fs.append(f_list[i][1])
    
# select the important features from top to bottom until the accumulated importance reaches 90%    
f_list2 = sorted(zip(map(lambda x: round(x, 4), importances/Sum), features), reverse=True)
Sum2 = 0
fs = []
for i in range(0, len(f_list2)):
    Sum2 = Sum2 + f_list2[i][0]
    fs.append(f_list2[i][1])
    if Sum2 >= 0.9:
        break
    
X_fs = df[fs].values

# Feature selection using FCBF
fcbf = FCBFK(k=20)
X_fss = fcbf.fit_transform(X_fs, y)

# Re-split train & test after featura selection
X_train, X_test, y_train, y_test = train_test_split(X_fss, y, train_size=0.8, test_size=0.2, random_state=0, stratify=y)

# Solve class-imbalance by SMOTE
smote = SMOTE(n_jobs=-1, sampling_strategy={2: 1000, 4: 1000})
X_train, y_train = smote.fit_resample(X_train, y_train)

# Machine learning model training
xg = xgb.XGBClassifier(n_estimators=10)
xg.fit(X_train, y_train)
xg_score = xg.score(X_test, y_test)
y_predict = xg.predict(X_test)
y_true = y_test
logging.info(f'Accuracy of XGBoost: {str(xg_score)}')
precision, recall, fscore, none = precision_recall_fscore_support(y_true, y_predict, average='weighted')
logging.info(f'Precision of XGBoost: {str(precision)}')
logging.info(f'Recall of XGBoost: {str(recall)}')
logging.info(f'F1-score of XGBoost: {str(fscore)}')
logging.info(f'Classification report of XGBoost: {classification_report(y_true, y_predict)}')
cm = confusion_matrix(y_true, y_predict)
f, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(cm, annot=True, linewidth=0.5, linecolor="red", fmt=".0f", ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.savefig("confusion_matrix_xgboost.png")  
plt.close() 

def objective(params):
    params = {
        'n_estimators': int(params['n_estimators']), 
        'max_depth': int(params['max_depth']),
        'learning_rate':  abs(float(params['learning_rate'])),

    }
    clf = xgb.XGBClassifier( **params)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test, y_pred)

    return {'loss':-score, 'status': STATUS_OK }

space = {
    'n_estimators': hp.quniform('n_estimators', 10, 100, 5),
    'max_depth': hp.quniform('max_depth', 4, 100, 1),
    'learning_rate': hp.normal('learning_rate', 0.01, 0.9),
}

best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=20)

logging.info("XGBoost: Hyperopt estimated optimum {}".format(best))

xg = xgb.XGBClassifier(learning_rate=best['learning_rate'], n_estimators=best['n_estimators'], max_depth=best['max_depth'])
xg.fit(X_train, y_train)
xg_score = xg.score(X_test, y_test)
y_predict = xg.predict(X_test)
y_true = y_test
logging.info(f'Accuracy of XGBoost: {str(xg_score)}')
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
logging.info(f'Precision of XGBoost: {str(precision)}')
logging.info(f'Recall of XGBoost: {str(recall)}')
logging.info(f'F1-score of XGBoost: {str(fscore)}')
logging.info(f'Classification report of XGBoost: {classification_report(y_true,y_predict)}')
cm=confusion_matrix(y_true,y_predict)
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.savefig("confusion_matrix_xgboost_hpo.png")  # Save the plot to a file
plt.close()

# Training Random Forest model
rf = RandomForestClassifier(random_state=0)
rf.fit(X_train, y_train)
rf_score = rf.score(X_test, y_test)
y_predict = rf.predict(X_test)
logging.info(f'Accuracy of RF: {str(rf_score)}')
precision, recall, fscore, none = precision_recall_fscore_support(y_true, y_predict, average='weighted')
logging.info(f'Precision of RF: {str(precision)}')
logging.info(f'Recall of RF: {str(recall)}')
logging.info(f'F1-score of RF: {str(fscore)}')
logging.info(f'Classification report of RF: {classification_report(y_true, y_predict)}')
cm = confusion_matrix(y_true, y_predict)
f, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")

plt.savefig("confusion_matrix_rf.png")  # Save the plot to a file
plt.close()

xg_train=xg.predict(X_train)
xg_test=xg.predict(X_test)

# Hyperparameter optimization for Random Forest
space_rf = {
    'n_estimators': hp.quniform('n_estimators', 10, 200, 1),
    'max_depth': hp.quniform('max_depth', 5, 50, 1),
    'max_features': hp.quniform('max_features', 1, 20, 1),
    'min_samples_split': hp.quniform('min_samples_split', 2, 11, 1),
    'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 11, 1),
    'criterion': hp.choice('criterion', ['gini', 'entropy'])
}

def objective_rf(params):
    params['n_estimators'] = int(params['n_estimators'])
    params['max_depth'] = int(params['max_depth'])
    params['max_features'] = int(params['max_features'])
    params['min_samples_split'] = int(params['min_samples_split'])
    params['min_samples_leaf'] = int(params['min_samples_leaf'])
    clf = RandomForestClassifier(**params)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    
    return {'loss': -score, 'status': STATUS_OK}

best_rf = fmin(fn=objective_rf, space=space_rf, algo=tpe.suggest, max_evals=20)
logging.info(f"Random Forest: Hyperopt estimated optimum {best_rf}")

rf_hpo = RandomForestClassifier(n_estimators = 71, min_samples_leaf = 1, max_depth = 46, min_samples_split = 9, max_features = 20, criterion = 'entropy')
rf_hpo.fit(X_train,y_train)
rf_score=rf_hpo.score(X_test,y_test)
y_predict=rf_hpo.predict(X_test)
y_true=y_test
logging.info(f'Accuracy of RF: {str(rf_score)}')
precision, recall, fscore, none = precision_recall_fscore_support(y_true, y_predict, average='weighted')
logging.info(f'Precision of RF: {str(precision)}')
logging.info(f'Recall of RF: {str(recall)}')
logging.info(f'F1-score of RF: {str(fscore)}')
logging.info(f'Classification report of RF: {classification_report(y_true, y_predict)}')
cm = confusion_matrix(y_true, y_predict)
f, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")

plt.savefig("confusion_matrix_rf_hpo.png")  # Save the plot to a file
plt.close()
rf_train=rf_hpo.predict(X_train)
rf_test=rf_hpo.predict(X_test)

# Training Decision Tree model
dt = DecisionTreeClassifier(random_state=0)
dt.fit(X_train, y_train)
dt_score = dt.score(X_test, y_test)
y_predict = dt.predict(X_test)
logging.info(f'Accuracy of DT: {str(dt_score)}')
logging.info(f'Classification report of DT: {classification_report(y_test, y_predict)}')
precision, recall, fscore, none = precision_recall_fscore_support(y_true, y_predict, average='weighted')
logging.info(f'Precision of DT: {str(precision)}')
logging.info(f'Recall of DT: {str(recall)}')
logging.info(f'F1-score of DT: {str(fscore)}')
cm=confusion_matrix(y_true,y_predict)
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")

plt.savefig("confusion_matrix_dt.png")  # Save the plot to a file
plt.close()

# Hyperparameter optimization for Decision Tree
space_dt = {
    'max_depth': hp.quniform('max_depth', 5, 50, 1),
    'max_features': hp.quniform('max_features', 1, 20, 1),
    'min_samples_split': hp.quniform('min_samples_split', 2, 11, 1),
    'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 11, 1),
    'criterion': hp.choice('criterion', ['gini', 'entropy'])
}
def objective_dt(params):
    params['max_depth'] = int(params['max_depth'])
    params['max_features'] = int(params['max_features'])
    params['min_samples_split'] = int(params['min_samples_split'])
    params['min_samples_leaf'] = int(params['min_samples_leaf'])
    clf = DecisionTreeClassifier(**params)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    return {'loss': -score, 'status': STATUS_OK}

best_dt = fmin(fn=objective_dt, space=space_dt, algo=tpe.suggest, max_evals=20)
logging.info(f"Best parameters for DT: {best_dt}")

dt_hpo = DecisionTreeClassifier(min_samples_leaf = 2, max_depth = 47, min_samples_split = 3, max_features = 19, criterion = 'gini')
dt_hpo.fit(X_train,y_train)
dt_score=dt_hpo.score(X_test,y_test)
y_predict=dt_hpo.predict(X_test)
y_true=y_test
logging.info(f'Accuracy of DT: {str(dt_score)}')
precision, recall, fscore, none = precision_recall_fscore_support(y_true, y_predict, average='weighted')
logging.info(f'Precision of DT: {str(precision)}')
logging.info(f'Recall of DT: {str(recall)}')
logging.info(f'F1-score of DT: {str(fscore)}')
logging.info(f'Classification report of DT: {classification_report(y_true, y_predict)}')
cm=confusion_matrix(y_true,y_predict)
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")

plt.savefig("confusion_matrix_dt_hpo.png")  # Save the plot to a file
plt.close()

dt_train=dt_hpo.predict(X_train)
dt_test=dt_hpo.predict(X_test)

# Training Extra Trees model
et = ExtraTreesClassifier(random_state=0)
et.fit(X_train, y_train)
et_score = et.score(X_test, y_test)
y_predict = et.predict(X_test)
logging.info(f'Accuracy of ET: {str(et_score)}')
logging.info(f'Classification report of ET: {classification_report(y_test, y_predict)}')
precision, recall, fscore, none = precision_recall_fscore_support(y_true, y_predict, average='weighted')
logging.info(f'Precision of ET: {str(precision)}')
logging.info(f'Recall of ET: {str(recall)}')
logging.info(f'F1-score of ET: {str(fscore)}')
cm = confusion_matrix(y_true, y_predict)
f, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")

plt.savefig("confusion_matrix_et.png")  # Save the plot to a file
plt.close()

# Hyperparameter optimization for Extra Trees
space_et = {
    'n_estimators': hp.quniform('n_estimators', 10, 200, 1),
    'max_depth': hp.quniform('max_depth', 5, 50, 1),
    'max_features': hp.quniform('max_features', 1, 20, 1),
    'min_samples_split': hp.quniform('min_samples_split', 2, 11, 1),
    'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 11, 1),
    'criterion': hp.choice('criterion', ['gini', 'entropy'])
}
def objective_et(params):
    params['n_estimators'] = int(params['n_estimators'])
    params['max_depth'] = int(params['max_depth'])
    params['max_features'] = int(params['max_features'])
    params['min_samples_split'] = int(params['min_samples_split'])
    params['min_samples_leaf'] = int(params['min_samples_leaf'])
    clf = ExtraTreesClassifier(**params)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    return {'loss': -score, 'status': STATUS_OK}
best_et = fmin(fn=objective_et, space=space_et, algo=tpe.suggest, max_evals=20)
logging.info(f"Best parameters for ET: {best_et}")

et_hpo = ExtraTreesClassifier(n_estimators = 53, min_samples_leaf = 1, max_depth = 31, min_samples_split = 5, max_features = 20, criterion = 'entropy')
et_hpo.fit(X_train,y_train) 
et_score=et_hpo.score(X_test,y_test)
y_predict=et_hpo.predict(X_test)
y_true=y_test
logging.info(f'Accuracy of ET: {str(et_score)}')
precision, recall, fscore, none = precision_recall_fscore_support(y_true, y_predict, average='weighted')
logging.info(f'Precision of ET: {str(precision)}')
logging.info(f'Recall of ET: {str(recall)}')
logging.info(f'F1-score of ET: {str(fscore)}')
logging.info(f'Classification report of ET: {classification_report(y_true, y_predict)}')
cm=confusion_matrix(y_true,y_predict)
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")

plt.savefig("confusion_matrix_et_hpo.png")  # Save the plot to a file
plt.close()

et_train=et_hpo.predict(X_train)
et_test=et_hpo.predict(X_test)

# Stacking ensemble model
base_predictions_train = pd.DataFrame( {
    'DecisionTree': dt_train.ravel(),
        'RandomForest': rf_train.ravel(),
     'ExtraTrees': et_train.ravel(),
     'XgBoost': xg_train.ravel(),
    })

dt_train=dt_train.reshape(-1, 1)
et_train=et_train.reshape(-1, 1)
rf_train=rf_train.reshape(-1, 1)
xg_train=xg_train.reshape(-1, 1)
dt_test=dt_test.reshape(-1, 1)
et_test=et_test.reshape(-1, 1)
rf_test=rf_test.reshape(-1, 1)
xg_test=xg_test.reshape(-1, 1)

x_train = np.concatenate(( dt_train, et_train, rf_train, xg_train), axis=1)
x_test = np.concatenate(( dt_test, et_test, rf_test, xg_test), axis=1)

stk = xgb.XGBClassifier().fit(x_train, y_train)
y_predict = stk.predict(x_test)
y_true=y_test
stk_score=accuracy_score(y_true,y_predict)
logging.info('Accuracy of Stacking: '+ str(stk_score))
precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
logging.info('Precision of Stacking: '+(str(precision)))
logging.info('Recall of Stacking: '+(str(recall)))
logging.info('F1-score of Stacking: '+(str(fscore)))
logging.info('Classification report of Stacking: '+(classification_report(y_true,y_predict)))
cm=confusion_matrix(y_true,y_predict)
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.savefig("confusion_matrix_stacking.png")  # Save the plot to a file
plt.close()

# Hyperparameter optimization for Stacking Ensemble Model
space_stk = {
    'n_estimators': hp.quniform('n_estimators', 10, 100, 5),
    'max_depth': hp.quniform('max_depth', 4, 100, 1),
    'learning_rate': hp.normal('learning_rate', 0.01, 0.9),
}

def objective_stk(params):
    params['n_estimators'] = int(params['n_estimators'])
    params['max_depth'] = int(params['max_depth'])
    params['learning_rate'] = abs(float(params['learning_rate']))
    clf = xgb.XGBClassifier(**params)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    score = accuracy_score(y_test, y_pred)
    return {'loss': -score, 'status': STATUS_OK}

best_stk = fmin(fn=objective_stk, space=space_stk, algo=tpe.suggest, max_evals=20)

logging.info(f"Best parameters for Stacking Ensemble Model: {best_stk}")

xg = xgb.XGBClassifier(learning_rate=abs(best_stk['learning_rate']), n_estimators=best_stk['n_estimators'], max_depth=best_stk['max_depth'])
xg.fit(x_train, y_train)
xg_score = xg.score(x_test, y_test)
y_predict = xg.predict(x_test)
y_true = y_test
logging.info('Accuracy of XGBoost: ' + str(xg_score))
precision, recall, fscore, none = precision_recall_fscore_support(y_true, y_predict, average='weighted')
logging.info('Precision of XGBoost: ' + (str(precision)))
logging.info('Recall of XGBoost: ' + (str(recall)))
logging.info('F1-score of XGBoost: ' + (str(fscore)))
logging.info('Classification report of XGBoost: ' + (classification_report(y_true, y_predict)))
cm=confusion_matrix(y_true,y_predict)
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.savefig("confusion_matrix_xgboost_stacking.png")  # Save the plot to a file
plt.close()

# Anomaly-based IDS
# Generate datasets for unknown attack detection
df = pd.read_csv('./data/CICIDS2017_sample_km.csv')

df1 = df[df['Label'] != 5]
df1['Label'][df1['Label'] > 0] = 1
df1.to_csv('./data/CICIDS2017_sample_km_without_portscan.csv', index=0)

df2 = df[df['Label'] == 5]
df2['Label'][df2['Label'] == 5] = 1
df2.to_csv('./data/CICIDS2017_sample_km_portscan.csv', index=0)

# Read the generated datasets
df1 = pd.read_csv('./data/CICIDS2017_sample_km_without_portscan.csv')
df2 = pd.read_csv('./data/CICIDS2017_sample_km_portscan.csv')

features = df1.drop(['Label'], axis=1).dtypes[df1.dtypes != 'object'].index
df1[features] = df1[features].apply(lambda x: (x - x.mean()) / (x.std()))
df2[features] = df2[features].apply(lambda x: (x - x.mean()) / (x.std()))

df1 = df1.fillna(0)
df2 = df2.fillna(0)

df2p=df1[df1['Label']==0]
df2pp=df2p.sample(n=None, frac=1255/18225, replace=False, weights=None, random_state=None, axis=0)
df2=pd.concat([df2, df2pp])

df = df1.append(df2)

X = df.drop(['Label'],axis=1) .values
y = df.iloc[:, -1].values.reshape(-1,1)
y=np.ravel(y)
pd.Series(y).value_counts()

importances = mutual_info_classif(X, y)
# calculate the sum of importance scores
f_list = sorted(zip(map(lambda x: round(x, 4), importances), features), reverse=True)
Sum = 0
fs = []
for i in range(0, len(f_list)):
    Sum = Sum + f_list[i][0]
    fs.append(f_list[i][1])
    
# select the important features from top to bottom until the accumulated importance reaches 90%
f_list2 = sorted(zip(map(lambda x: round(x, 4), importances/Sum), features), reverse=True)
Sum2 = 0
fs = []
for i in range(0, len(f_list2)):
    Sum2 = Sum2 + f_list2[i][0]
    fs.append(f_list2[i][1])
    if Sum2>=0.9:
        break        
    
X_fs = df[fs].values

fcbf = FCBFK(k = 20)
X_fss = fcbf.fit_transform(X_fs,y)

from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components = 10, kernel = 'rbf')
kpca.fit(X_fss, y)
X_kpca = kpca.transform(X_fss)

X_train = X_kpca[:len(df1)]
y_train = y[:len(df1)]
X_test = X_kpca[len(df1):]
y_test = y[len(df1):]

# Solve class-imbalance by SMOTE
smote = SMOTE(n_jobs=-1, sampling_strategy={1: 18225})
X_train, y_train = smote.fit_resample(X_train, y_train)

# Apply the cluster labeling (CL) k-means method
def CL_kmeans(X_train, X_test, y_train, y_test, n, b=100):
    km_cluster = MiniBatchKMeans(n_clusters=n, batch_size=b)
    result = km_cluster.fit_predict(X_train)
    result2 = km_cluster.predict(X_test)
    a = np.zeros(n)
    b = np.zeros(n)
    for v in range(0, n):
        for i in range(0, len(y_train)):
            if result[i] == v:
                if y_train[i] == 1:
                    a[v] += 1
                else:
                    b[v] += 1
                    
    list1=[]
    list2=[]
    for v in range(0,n):
        if a[v]<=b[v]:
            list1.append(v)
        else: 
            list2.append(v)
            
    for v in range(0, len(y_test)):
        if result2[v] in list1:
            result2[v] = 0
        elif result2[v] in list2:
            result2[v] = 1
            
    logging.info(classification_report(y_test, result2))
    cm = confusion_matrix(y_test, result2)
    acc = accuracy_score(y_test, result2)
    logging.info(str(acc))
    logging.info(cm)

CL_kmeans(X_train, X_test, y_train, y_test, 8)

#Hyperparameter optimization by BO-GP

space  = [Integer(2, 50, name='n_clusters')]
@use_named_args(space)
def objective(**params):
    km_cluster = MiniBatchKMeans(batch_size=100, **params)
    n=params['n_clusters']
    
    result = km_cluster.fit_predict(X_train)
    result2 = km_cluster.predict(X_test)

    count=0
    a=np.zeros(n)
    b=np.zeros(n)
    for v in range(0,n):
        for i in range(0,len(y_train)):
            if result[i]==v:
                if y_train[i]==1:
                    a[v]=a[v]+1
                else:
                    b[v]=b[v]+1
    list1=[]
    list2=[]
    for v in range(0,n):
        if a[v]<=b[v]:
            list1.append(v)
        else: 
            list2.append(v)
    for v in range(0,len(y_test)):
        if result2[v] in list1:
            result2[v]=0
        elif result2[v] in list2:
            result2[v]=1
        else:
            print("-1")
    cm=metrics.accuracy_score(y_test,result2)
    logging.info(str(n)+" "+str(cm))
    return (1-cm)

from skopt import gp_minimize
import time
t1=time.time()
res_gp = gp_minimize(objective, space, n_calls=20, random_state=0)
t2=time.time()
logging.info(t2-t1)
logging.info("Best score=%.4f" % (1-res_gp.fun))
logging.info("""Best parameters: n_clusters=%d""" % (res_gp.x[0]))

#Hyperparameter optimization by BO-TPE
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.cluster import MiniBatchKMeans
from sklearn import metrics

def objective(params):
    params = {
        'n_clusters': int(params['n_clusters']), 
    }
    km_cluster = MiniBatchKMeans(batch_size=100, **params)
    n=params['n_clusters']
    
    result = km_cluster.fit_predict(X_train)
    result2 = km_cluster.predict(X_test)

    count=0
    a=np.zeros(n)
    b=np.zeros(n)
    for v in range(0,n):
        for i in range(0,len(y_train)):
            if result[i]==v:
                if y_train[i]==1:
                    a[v]=a[v]+1
                else:
                    b[v]=b[v]+1
    list1=[]
    list2=[]
    for v in range(0,n):
        if a[v]<=b[v]:
            list1.append(v)
        else: 
            list2.append(v)
    for v in range(0,len(y_test)):
        if result2[v] in list1:
            result2[v]=0
        elif result2[v] in list2:
            result2[v]=1
        else:
            print("-1")
    score=metrics.accuracy_score(y_test,result2)
    print(str(params['n_clusters'])+" "+str(score))
    return {'loss':1-score, 'status': STATUS_OK }
space = {
    'n_clusters': hp.quniform('n_clusters', 2, 50, 1),
}

best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=20)
logging.info("Random Forest: Hyperopt estimated optimum {}".format(best))

CL_kmeans(X_train, X_test, y_train, y_test, 16) 
