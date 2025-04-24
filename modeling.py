from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import confusion_matrix, accuracy_score
from collections import Counter

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

import numpy as np
import pandas as pd

def spectral(d, train_cols):
    clusters = []
    accuracy = []
    
    clustering = SpectralClustering(n_clusters = 2, gamma = 1, random_state = 0).fit_predict(d[train_cols])
    accuracy = accuracy_score(d['Review Class'], clustering)

    cm = confusion_matrix(d['Review Class'], clustering)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print(cm)

    print('Precision 0: %.2f%%' % (cm[0][0]/sum(cm[0]) * 100.0))
    print('Precision 1: %.2f%%' % (cm[1][1]/sum(cm[1]) * 100.0))

    print('Recall 0: %.2f%%' % (cm[0][0]/(cm[0][0] + cm[1][0]) * 100.0))
    print('Recall 1: %.2f%%' % (cm[1][1]/(cm[0][1] + cm[1][1])  * 100.0))
    print('\n')
    
def kmeans(d, train_cols):
    kmeans = KMeans(n_clusters = 2, random_state = 0).fit_predict(np.array(d[train_cols]))

    accuracy = accuracy_score(d['Review Class'], kmeans)

    cm = confusion_matrix(d['Review Class'], kmeans)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print(cm)

    print('Precision 0: %.2f%%' % (cm[0][0]/sum(cm[0]) * 100.0))
    print('Precision 1: %.2f%%' % (cm[1][1]/sum(cm[1]) * 100.0))

    print('Recall 0: %.2f%%' % (cm[0][0]/(cm[0][0] + cm[1][0]) * 100.0))
    print('Recall 1: %.2f%%' % (cm[1][1]/(cm[0][1] + cm[1][1])  * 100.0))
    print('\n')
    
def rfm(d, train_cols):
    Y = d['Review Class']
    X = d[train_cols]
    Y = Y.astype(float)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=35)

    rf = RandomForestClassifier(random_state = 0)
    rf.fit(X_train, Y_train)

    Y_pred_rf = rf.predict(X_test)
    
    accuracy = accuracy_score(Y_test, Y_pred_rf)
    cm = confusion_matrix(Y_test, Y_pred_rf)

    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print(cm)

    print('Precision 0: %.2f%%' % (cm[0][0]/sum(cm[0]) * 100.0))
    print('Precision 1: %.2f%%' % (cm[1][1]/sum(cm[1]) * 100.0))

    print('Recall 0: %.2f%%' % (cm[0][0]/(cm[0][0] + cm[1][0]) * 100.0))
    print('Recall 1: %.2f%%' % (cm[1][1]/(cm[0][1] + cm[1][1])  * 100.0))
    print('\n')
    
    return rf
    
def xgbm(d, train_cols):
    Y = d['Review Class']
    X = d[train_cols]
    Y = Y.astype(float)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=35)
    evalset = [(X_train, Y_train), (X_test,Y_test)]


    counter = Counter(Y_train)
    estimate = counter[0]/counter[1]

    xgbc = xgb.XGBClassifier(n_estimators=250, eta=0.01, scale_pos_weight = estimate, eval_metric = ['error', 'logloss'], importance_type = 'weight')
    xgbc.fit(X_train, Y_train, eval_set = evalset, verbose = False)

    Y_pred_xgb = xgbc.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred_xgb)
    cm = confusion_matrix(Y_test, Y_pred_xgb)

    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print(cm)

    print('Precision 0: %.2f%%' % (cm[0][0]/sum(cm[0]) * 100.0))
    print('Precision 1: %.2f%%' % (cm[1][1]/sum(cm[1]) * 100.0))

    print('Recall 0: %.2f%%' % (cm[0][0]/(cm[0][0] + cm[1][0]) * 100.0))
    print('Recall 1: %.2f%%' % (cm[1][1]/(cm[0][1] + cm[1][1])  * 100.0))
    print('\n')
    
    return xgbc