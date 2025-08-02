import subprocess
import zipfile
import os
import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder

subprocess.run(['wget', 'http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/CSVs/MachineLearningCSV.zip', '-O', 'CIC_IDS_2017.zip'], check=True) 

with zipfile.ZipFile("CIC_IDS_2017.zip", 'r') as zip_ref:
    zip_ref.extractall("./data")
    
df_list = []
for file in os.listdir('./data/MachineLearningCVE/'):
  df_aux = pd.read_csv(f'./data/MachineLearningCVE/{file}')
  df_list.append(df_aux)
  
df = pd.concat(df_list, ignore_index=True)

df.columns = df.columns.str.strip()

mapping = {
    'DoS Hulk': 'DoS',
    'FTP-Patator': 'BruteForce',
    'SSH-Patator': 'BruteForce',
    'DoS GoldenEye': 'DoS',
    'DoS slowloris': 'DoS',
    'DoS Slowhttptest': 'DoS',
    'Web Attack � Brute Force': 'WebAttack',
    'Web Attack � XSS': 'WebAttack',
    'Web Attack � Sql Injection': 'WebAttack',
    'DDoS': 'DoS',
    'Heartbleed': 'DoS'
}

df['Label'] = df['Label'].replace(mapping)

df = df.fillna(0)

# Z-score normalization
features = df.dtypes[df.dtypes != 'object'].index
df[features] = df[features].apply(
    lambda x: (x - x.mean()) / (x.std()))

# Fill empty values by 0
df = df.fillna(0)

labelencoder = LabelEncoder()
df.iloc[:, -1] = labelencoder.fit_transform(df.iloc[:, -1])

# retain the minority class instances and sample the majority class instances
df_minor = df[(df['Label']==6)|(df['Label']==1)|(df['Label']==4)]
df_major = df.drop(df_minor.index)

X = df_major.drop(['Label'],axis=1) 
y = df_major.iloc[:, -1].values.reshape(-1,1)
y=np.ravel(y)

# use k-means to cluster the data samples and select a proportion of data from each cluster
from sklearn.cluster import MiniBatchKMeans
kmeans = MiniBatchKMeans(n_clusters=1000, random_state=0).fit(X)

klabel=kmeans.labels_
df_major['klabel']=klabel

cols = list(df_major)
cols.insert(78, cols.pop(cols.index('Label')))
df_major = df_major.loc[:, cols]

def typicalSampling(group):
    name = group.name
    frac = 0.008
    return group.sample(frac=frac)

result = df_major.groupby(
    'klabel', group_keys=False
).apply(typicalSampling)

result = result.drop(['klabel'],axis=1)
result = pd.concat([result, df_minor], ignore_index=True)

result.to_csv('./data/CICIDS2017_sample_km.csv',index=0)