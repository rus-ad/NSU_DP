# ## Tf-Idf

import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import pandas as pd
random_state = 7

features = {}
base_path = '../parser/data/'
for foldef_name in os.listdir(base_path):
    filenames = os.listdir(base_path + foldef_name)
    if filenames:
        features[foldef_name] = ' '.join(filenames).replace('.mp3', '')

vectorizer = TfidfVectorizer(analyzer='char_wb', max_features=100, ngram_range=(3,3))
X = vectorizer.fit_transform(list(features.values()))
pca = PCA(n_components=2)
X_compact = pca.fit_transform(X.toarray())
model = KMeans(n_clusters=3, random_state=random_state)
y_pred = model.fit_predict(X)

df = pd.DataFrame(dict(name=features.keys(), songs=features.values()))
df['predict'] = y_pred

leaders = []
for center in model.cluster_centers_:
    distances = []
    for vector in X.toarray():
        distances.append(np.linalg.norm(vector - center))
        idx = set(range(len(distances))) - set(leaders)
    leaders.append(np.argmin(np.array(distances)[list(idx)]))

plt.figure(figsize=(20, 20))
plt.scatter(X_compact[:, 0], X_compact[:, 1], c=y_pred)
plt.scatter(X_compact[leaders, 0], X_compact[leaders, 1], s=180, marker='*')
plt.title("KMeans");

from collections import Counter
import re

test = ' '.join(df[df['predict'] == 1]['songs'].tolist())
Counter([i for i in re.split('_|\s', test) if len(i) > 2]).most_common(30)

test = ' '.join(df[df['predict'] == 2]['songs'].tolist())
Counter([i for i in re.split('_|\s', test) if len(i) > 2]).most_common(30)

test = ' '.join(df[df['predict'] == 0]['songs'].tolist())
Counter([i for i in re.split('_|\s', test) if len(i) > 2]).most_common(30)

vectorizer = TfidfVectorizer(analyzer='word', max_features=1000)
X = vectorizer.fit_transform(list(features.values()))
pca = PCA(n_components=2)
X_compact = pca.fit_transform(X.toarray())
model = DBSCAN(eps=0.05, min_samples=2)
y_pred = model.fit_predict(X)

df = pd.DataFrame(dict(name=features.keys(), songs=features.values()))
df['predict'] = y_pred

plt.figure(figsize=(12, 12))
plt.scatter(X_compact[:, 0], X_compact[:, 1], c=y_pred)
plt.title("DBSCAN");

features_df = pd.read_pickle('../parser/save.pkl')

for idx, i in enumerate(features_df['spec_cent'].to_list()):
    if i.shape != (1, 1292):
        if i.shape[1] > 1292:
            features_df['spec_cent'].iloc[idx] = i[:, :1292]
            continue
        
        features_df['spec_cent'].iloc[idx] = np.pad(
            i.flatten(), 
            (0, 1292 - i.shape[1]), 
            'constant', 
            constant_values=(0, 0)
        ).reshape(1, 1292)

result = {}
for name in set(features_df['name']):
    if name not in result:
        result[name] = []
    result[name].append(
        np.mean(features_df[features_df['name'] == name]['spec_cent'].tolist(), 
                axis=0).flatten()
    )
train = pd.DataFrame(result).T

X = train[0].tolist()
pca = PCA(n_components=2)
X_compact = pca.fit_transform(X)
model = KMeans(n_clusters=3, random_state=random_state)
y_pred = model.fit_predict(X)

train['spec_cent_predict'] = y_pred

leaders = []
for center in model.cluster_centers_:
    distances = []
    for vector in X:
        distances.append(np.linalg.norm(vector - center))
        idx = set(range(len(distances))) - set(leaders)
    leaders.append(np.argmin(np.array(distances)[list(idx)]))

plt.figure(figsize=(20, 20))
plt.scatter(X_compact[:, 0], X_compact[:, 1], c=y_pred)
plt.scatter(X_compact[leaders, 0], X_compact[leaders, 1], s=180, marker='*')
plt.title("KMeans");

train['songs'] = train.index.map(df.set_index('name')['songs'])

test = ' '.join(train[train['spec_cent_predict'] == 0]['songs'].tolist())
Counter([i for i in re.split('_|\s', test) if len(i) > 2]).most_common(30)

test = ' '.join(train[train['spec_cent_predict'] == 1]['songs'].tolist())
Counter([i for i in re.split('_|\s', test) if len(i) > 2]).most_common(30)

test = ' '.join(train[train['spec_cent_predict'] == 2]['songs'].tolist())
Counter([i for i in re.split('_|\s', test) if len(i) > 2]).most_common(30)

# +
for idx, i in enumerate(features_df['mfcc'].to_list()):
    if i.shape != (1, 1292):
        if i.shape[1] > 1292:
            features_df['mfcc'].iloc[idx] = i[:, :1292]
            continue
        
        features_df['mfcc'].iloc[idx] = np.pad(
            i, 
            [(0, 0), (0, 1292 - i.shape[1])], 
            'constant', 
            constant_values=(0, 0)
        ).reshape(20, 1292)
        
result = {}
for name in set(features_df['name']):
    if name not in result:
        result[name] = []
    result[name].append(
        np.mean(features_df[features_df['name'] == name]['mfcc'].tolist(), 
                axis=0).flatten()
    )
train = pd.DataFrame(result).T
train['songs'] = train.index.map(df.set_index('name')['songs'])
# -

X = train[0].tolist()
pca = PCA(n_components=2)
X_compact = pca.fit_transform(X)
model = KMeans(n_clusters=2, random_state=random_state)
y_pred = model.fit_predict(X)
train['mfcc_predict'] = y_pred

leaders = []
for center in model.cluster_centers_:
    distances = []
    for vector in X:
        distances.append(np.linalg.norm(vector - center))
        idx = set(range(len(distances))) - set(leaders)
    leaders.append(np.argmin(np.array(distances)[list(idx)]))

plt.figure(figsize=(20, 20))
plt.scatter(X_compact[:, 0], X_compact[:, 1], c=y_pred)
plt.scatter(X_compact[leaders, 0], X_compact[leaders, 1], s=180, marker='*')
plt.title("KMeans");

train['songs'] = train.index.map(df.set_index('name')['songs'])
test = ' '.join(train[train['mfcc_predict'] == 0]['songs'].tolist())
Counter([i for i in re.split('_|\s', test) if len(i) > 2]).most_common(30)

test = ' '.join(train[train['mfcc_predict'] == 1]['songs'].tolist())
Counter([i for i in re.split('_|\s', test) if len(i) > 2]).most_common(30)

for idx, i in enumerate(features_df['spec_bw'].to_list()):
    if i.shape != (1, 1292):
        if i.shape[1] > 1292:
            features_df['spec_bw'].iloc[idx] = i[:, :1292]
            continue
        
        features_df['spec_bw'].iloc[idx] = np.pad(
            i.flatten(), 
            (0, 1292 - i.shape[1]), 
            'constant', 
            constant_values=(0, 0)
        ).reshape(1, 1292)

for idx, i in enumerate(features_df['rolloff'].to_list()):
    if i.shape != (1, 1292):
        if i.shape[1] > 1292:
            features_df['rolloff'].iloc[idx] = i[:, :1292]
            continue
        
        features_df['rolloff'].iloc[idx] = np.pad(
            i.flatten(), 
            (0, 1292 - i.shape[1]), 
            'constant', 
            constant_values=(0, 0)
        ).reshape(1, 1292)

for idx, i in enumerate(features_df['zcr'].to_list()):
    if i.shape != (1, 1292):
        if i.shape[1] > 1292:
            features_df['zcr'].iloc[idx] = i[:, :1292]
            continue
        
        features_df['zcr'].iloc[idx] = np.pad(
            i.flatten(), 
            (0, 1292 - i.shape[1]), 
            'constant', 
            constant_values=(0, 0)
        ).reshape(1, 1292)

for idx, i in enumerate(features_df['rmse'].to_list()):
    if i.shape != (1, 1292):
        if i.shape[1] > 1292:
            features_df['rmse'].iloc[idx] = i[:, :1292]
            continue
        
        features_df['rmse'].iloc[idx] = np.pad(
            i.flatten(), 
            (0, 1292 - i.shape[1]), 
            'constant', 
            constant_values=(0, 0)
        ).reshape(1, 1292)

for idx, i in enumerate(features_df['chroma_stft'].to_list()):
    if i.shape != (1, 1292):
        if i.shape[1] > 1292:
            features_df['chroma_stft'].iloc[idx] = i[:, :1292]
            continue
        
        features_df['chroma_stft'].iloc[idx] = np.pad(
            i, 
            [(0, 0), (0, 1292 - i.shape[1])], 
            'constant', 
            constant_values=(0, 0)
        ).reshape(12, 1292)

# +
result = {}
for name in set(features_df['name']):
    if name not in result:
        result[name] = []
    mfcc = np.mean(features_df[features_df['name'] == name]['mfcc'].tolist()).flatten()[0]
    spec_cent = np.mean(features_df[features_df['name'] == name]['spec_cent'].tolist()).flatten()[0]
    spec_bw = np.mean(features_df[features_df['name'] == name]['spec_bw'].tolist()).flatten()[0]
    rolloff = np.mean(features_df[features_df['name'] == name]['rolloff'].tolist()).flatten()[0]
    zcr = np.mean(features_df[features_df['name'] == name]['zcr'].tolist()).flatten()[0]
    rmse = np.mean(features_df[features_df['name'] == name]['rmse'].tolist()).flatten()[0]
    chroma_stft = np.mean(features_df[features_df['name'] == name]['chroma_stft'].tolist()).flatten()[0]
    result[name].append([mfcc, spec_cent, spec_bw, rolloff, zcr, rmse, chroma_stft])

train = pd.DataFrame(result).T
train['songs'] = train.index.map(df.set_index('name')['songs'])
# -

X = train[0].tolist()
pca = PCA(n_components=2)
X_compact = pca.fit_transform(X)
model = KMeans(n_clusters=3, random_state=random_state)
y_pred = model.fit_predict(X)
train['all_predict'] = y_pred

leaders = []
for center in model.cluster_centers_:
    distances = []
    for vector in X:
        distances.append(np.linalg.norm(vector - center))
        idx = set(range(len(distances))) - set(leaders)
    leaders.append(np.argmin(np.array(distances)[list(idx)]))

plt.figure(figsize=(20, 20))
plt.scatter(X_compact[:, 0], X_compact[:, 1], c=y_pred)
plt.scatter(X_compact[leaders, 0], X_compact[leaders, 1], s=180, marker='*')
plt.title("KMeans");

test = ' '.join(train[train['all_predict'] == 0]['songs'].tolist())
Counter([i for i in re.split('_|\s', test) if len(i) > 2]).most_common(30)

test = ' '.join(train[train['all_predict'] == 1]['songs'].tolist())
Counter([i for i in re.split('_|\s', test) if len(i) > 2]).most_common(30)

test = ' '.join(train[train['all_predict'] == 2]['songs'].tolist())
Counter([i for i in re.split('_|\s', test) if len(i) > 2]).most_common(30)



# ## GraphSAGE

# +
import networkx as nx
import pandas as pd
import numpy as np
import os
import random

import stellargraph as sg
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import GraphSAGELinkGenerator
from stellargraph.layer import GraphSAGE, link_classification
from stellargraph.data import UniformRandomWalk
from stellargraph.data import UnsupervisedSampler
from sklearn.model_selection import train_test_split

from tensorflow import keras
from sklearn import preprocessing, feature_extraction, model_selection
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import accuracy_score

from stellargraph import globalvar

from stellargraph import datasets
from IPython.display import display, HTML
# -

vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,3))
X = vectorizer.fit_transform(list(features.values()))

G = StellarGraph(X.toarray())

dataset = datasets.Cora()
display(HTML(dataset.description))
G, node_subjects = dataset.load()

nodes = list(G.nodes())
number_of_walks = 1
length = 5
unsupervised_samples = UnsupervisedSampler(
    G, nodes=nodes, length=length, number_of_walks=number_of_walks
)

batch_size = 50
epochs = 4
num_samples = [10, 5]

generator = GraphSAGELinkGenerator(G, batch_size, num_samples)
train_gen = generator.flow(unsupervised_samples)



# +
# feature extractoring and preprocessing data
import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import os
from PIL import Image
import pathlib
import csv

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

#Keras
import keras
# -
file = open('test.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
for g in genres:
    for filename in os.listdir(f'{path}{g}'):
        songname = f'{path}{g}/{filename}'
        y, sr = librosa.load(songname, mono=True, duration=30)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        rmse = librosa.feature.rms(y=y)
        to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        to_append += f' {g}'
        file = open('test.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())



cls = keras.models.load_model('cls_genre')















