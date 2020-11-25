# +
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
random_state = 7

corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())

print(X.shape)

# +
y_pred = KMeans(n_clusters=2, random_state=random_state).fit_predict(X)

plt.figure(figsize=(12, 12))
plt.scatter(X.toarray()[:, 0], X.toarray()[:, 1], c=y_pred)
plt.title("KMeans")
# -

clustering = DBSCAN(eps=1, min_samples=2).fit_predict(X)

plt.figure(figsize=(12, 12))
plt.scatter(X.toarray()[:, 0], X.toarray()[:, 1], c=clustering)
plt.title("DBSCAN")










