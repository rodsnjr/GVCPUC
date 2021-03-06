# coding=utf-8
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.cluster.vq import vq
# Validations
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics

class BOW():
    def __init__(self):
        self.n_clusters = 20
        self.kmeans = KMeans(n_clusters=self.n_clusters)
        self.vocabulary = []
        self.clf = SVC()

    def train(self, x, y):
        self.k_dict = self.kmeans.fit_predict(x)
        self.__vocabulary(x, len(x))
        self.__standardize()
        self.clf.fit(self.vocabulary, y)

    def train_flow(self, flow):
        self.train(flow.X(), flow.Y())

    def evaluate(self, x, y):
        # vocabX = np.array([self.kmeans.predict(val) for val in x])
        histX = np.array([self.__computeHistogram(val) for val in x])
        return self.clf.score(histX, y)
    
    def cross_val(self, x, y):
        histX = np.array([self.__computeHistogram(val) for val in x])
        predicted = cross_val_predict(self.clf, histX, y, cv=10)
        return metrics.accuracy_score(y, predicted)
    
    def histX(self, x):
        return np.array([self.__computeHistogram(val) for val in x])

    def evaluate_flow(self, flow):
        return self.evaluate(flow.X(), flow.Y())

    def __vocabulary(self, x, size):
        self.vocabulary = []
        for f in x:
            histogram = self.__computeHistogram(f)
            self.vocabulary.append(histogram)
        self.vocabulary = np.array(self.vocabulary)
    
    def __computeHistogram(self, x):
        code, dist = vq(x, self.k_dict)
        bins = range(x.shape[0] + 1)
        histogram, edges = np.histogram(code, bins=bins, normed=True)
        return histogram

    def __standardize(self):
        self.scale = StandardScaler().fit(self.vocabulary)
        self.vocabulary = self.scale.transform(self.vocabulary)
