#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import scipy as sp
from scipy.spatial.distance import mahalanobis
from scipy.spatial.distance import cdist
from matplotlib.colors import ListedColormap
from sklearn import preprocessing, cross_validation, neighbors,datasets
from matplotlib.patches import Ellipse
import scipy.io as sc
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel 
import seaborn as sns 

import pandas as pd
from pandas import read_csv
import numpy as np
from  itertools import combinations
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from scipy import interp
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz 
import matplotlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier 
from sklearn.metrics import precision_recall_fscore_support as score
from time import time
from sklearn.naive_bayes import GaussianNB

from keras.models import Sequential
from keras.layers import Dense, Dropout

import scipy.io
c5=pd.read_csv('E:/waterloo/term2/data_modelling/Project/creditcardfraud/creditcard.csv')

#####This data is the splice junctions on DNA sequences.
##The given dataset includes 2200 samples with 57 features, in the
##matrix 'fea'. It is a binary class problem. The class labels are either +1 or -1, given in the vector 'gnd'.
## Parameter selection and classification tasks are conducted on this dataset.


 

#####KNN

#X=np.vstack((ca3,cb3))
X=c5.drop('Class',axis=1)

#x_n4=X[0:int(len(X)/4)]
#X=x_n4
#label=np.zeros(len(ca3)+len(cb3))
#label[0:200]=1
#label[200:len(label)]=2
label=c5['Class']
y=label
# fit an Extra Trees model to the data
model = ExtraTreesClassifier()
model.fit(X_res, y_res)
# display the relative importance of each attribute
print(model.feature_importances_)
a=model.feature_importances_


yy=pd.DataFrame(label)
yn=(yy == 0).astype(int).sum()
yp=(yy == 1).astype(int).sum()



## SMOTE
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE 
#X, y = make_classification(n_classes=2, class_sep=2,
#                           weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
#                           n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
print('Original dataset shape {}'.format(Counter(y)))

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_sample(X, y)
print('Resampled dataset shape {}'.format(Counter(y_res)))



## ADASYN
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import ADASYN 
#X, y = make_classification(n_classes=2, class_sep=2,
#                           weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
#                           n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
print('Original dataset shape {}'.format(Counter(y)))

sm = ADASYN(random_state=42)
X_res, y_res = sm.fit_sample(X, y)
print('Resampled dataset shape {}'.format(Counter(y_res)))


 

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2)


#n_neighbors = 5
from sklearn.cross_validation import cross_val_score, cross_val_predict

best_score=[]
acc=[]
besta_val=[]
k=np.arange(1,32,2)
X = X_train
y = y_train

#plt.figure(1)
#ax = sns.countplot(y,label='count')
#plt.xlabel('class -1 and 1')
#plt.title('class distribution')


for i in range(1,32,2):



    clf = neighbors.KNeighborsClassifier(3, weights='distance')
    clf.fit(X, y)
    knn_predict = clf.predict(X_test)
    accuracy=accuracy_score(y_test, knn_predict)
    acc.append(accuracy)
    
    
    #X = ["a", "b", "c", "d"]
    #kf = KFold(n_splits=5)
    #for train in kf.split(X):
    #    print("%s %s" % (train))
    
    
    # 10-fold cross-validation with K=5 for KNN (the n_neighbors parameter)
    # k = 5 for KNeighborsClassifier
    #knn = KNeighborsClassifier(n_neighbors=5)
    # Use cross_val_score function
    # We are passing the entirety of X and y, not X_train or y_train, it takes care of splitting the dat
    # cv=10 for 10 folds
    # scoring='accuracy' for evaluation metric - althought they are many
#    X=c5['fea']
#    y = label.T[0]
    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    predicted = cross_val_predict(clf, X, y, cv=5)
    besta_val.append(np.max(scores))
    print(scores)
    best_score.append(scores)

highest=np.argwhere(best_score==np.max(best_score))
best_k=k[highest[0][0]]
matplotlib.rcParams.update({'font.size': 22})
plt.figure()
plt.scatter(k,besta_val)
plt.xlabel('k Value')
plt.ylabel('Accuracy')
plt.title('Plot showing Kvalue verses accuracy')

report = classification_report(y_test,knn_predict)
fpr, tpr, thresholds = roc_curve(y_test, knn_predict)
roc_auc = auc(fpr, tpr)
#            aucs=np.append(aucs,roc_auc,axis=0)

print("ROC_AUC of KNN :", roc_auc)

##########################################################################################################################



#'For K=6 : 0.784'
#k=2 , 0.74
#k=3, 0.736
#k=4, 0.702,



#########SVM
    



def sigmatogamma(sig):
    gamma=1/(2*np.square(sig))
    return gamma



c_set=[0.1, 0.5, 1, 2, 5, 10, 20, 50]
sig_set=[0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]   
gamma_set=sigmatogamma(sig_set)
gamma_set=np.ndarray.tolist(gamma_set)

fpr = []
tpr = []
auc_stack=[]
cv = StratifiedKFold(n_splits=5)
fpr_stack=[]
tpr_stack=[]
X=c5['fea']

y=c5['gnd']
    
aucs = [] 
for c in c_set:
    for g in gamma_set:
#        tprs = []
        
#        mean_fpr = np.linspace(0, 1, 100)
        i = 0
        rbf_svc = svm.SVC(C=c,kernel='rbf',gamma=g)
        for train, test in cv.split(X, y):
#            rbf_svc = svm.SVC(C=c,kernel='rbf',gamma=g,probability=True)
#            probas_ = rbf_svc.fit(X[train], y[train]).predict_proba(X[test])
#            # Compute ROC curve and area the curve
#            fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
#            tprs.append(interp(mean_fpr, fpr, tpr))
#            tprs[-1][0] = 0.0
#            roc_auc = auc(fpr, tpr)
##            np.append(auc_stack,roc_auc,axis=0)
#            aucs.append(roc_auc,axis=0)
#            plt.plot(fpr, tpr, lw=1, alpha=0.3,
#                     label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
#        
#            i += 1
            
            
#            rbf_svc.fit(X_train, y_train)
            y_score = rbf_svc.fit(X[train], y[train]).decision_function(X[test])
#            svm_predict=rbf_svc.predict(X_test)
#            predicted = cross_val_predict(rbf_svc, X, y, cv=5)
#            fpr[c_set.index(c),gamma_set.index(g),i], tpr[c_set.index(c),gamma_set.index(g),i], _ = roc_curve(y[test], y_score)
            fpr, tpr, _ = roc_curve(y[test], y_score)
            fpr_stack.append(fpr)
            tpr_stack.append(tpr)

            roc_auc = auc(fpr, tpr)
#            aucs=np.append(aucs,roc_auc,axis=0)
            aucs.append(roc_auc)
            i=i+1
            
                
        
au=np.reshape(aucs,[8,8,5])        
highest_auc=np.argwhere(au==np.max(au)) 
highest_auc=np.ndarray.tolist(highest_auc)       



#rbf_svc = svm.SVC(C=c,kernel='rbf',gamma=g)
#rbf_svc.fit(X_train, y_train)
#y_score = rbf_svc.fit(X_train, y_train).decision_function(X_test)
#svm_predict=rbf_svc.predict(X_test)
#predicted = cross_val_predict(rbf_svc, X, y, cv=5)
#fpr, tpr, _ = roc_curve(y_test, predicted)
#roc_auc = auc(fpr, tpr)
#auc.append(roc_auc)

#accuracy_score(y_test, svm_predict)
for i in range(len(highest_auc)):
    
    plt.figure()
    lw = 2
    acc=au[highest_auc[i][0]][highest_auc[i][1]][highest_auc[i][2]]
    plt.plot(fpr_stack[highest_auc[i][0]*highest_auc[i][1]*highest_auc[i][2] -1], tpr_stack[highest_auc[i][0]*highest_auc[i][1]*highest_auc[i][2] -1], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' )
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example' )
    plt.legend(loc="lower right")
    plt.show()

## 2(3)
from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support) 
sample_leaf_options = [1,5,10,50,100,200,500]
   
RF = RandomForestClassifier(min_samples_split=20, random_state=99,max_depth=(len(X_train)-1))
RF.fit(X_train, y_train) 
preditct_RF = RF.predict(X_test)
print('accuracy using RF:',accuracy_score(preditct_RF, y_test))

sm = svm.SVC(C=5,kernel='rbf',gamma=0.02)
sm.fit(X_train, y_train) 
preditct_sm = sm.predict(X_test)
print('accuracy using sm:',accuracy_score(preditct_sm, y_test))
### MLP
mlp_clf = MLPClassifier(solver='sgd', alpha=1e-4,hidden_layer_sizes=(10,3),learning_rate='adaptive', random_state=1,activation='tanh')
mlp_clf.fit(X_train, y_train)
preditct_mlp = mlp_clf.predict(X_test)
print('accuracy using NN:',accuracy_score(preditct_mlp, y_test))  

report = classification_report(y_test,preditct_RF)
fpr, tpr, thresholds = roc_curve(y_test, preditct_RF)
roc_auc = auc(fpr, tpr)
## MLP model 2
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(30,)))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='tanh'))
model.add(Dense(2, activation='softmax'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
history = model.fit(X_train, y_train,
                    batch_size=500,
                    epochs=10,
                    verbose=1,
                    validation_data=(X_test, y_test))


plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, label='AUC = %0.4f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
#plt.xlim([-0.001, 1])
#plt.ylim([0, 1.001])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show();

##3c
cv = StratifiedKFold(n_splits=5)

precision_svm=[]
recall_svm=[]
fscore_svm=[]
accuracy_svm=[]
train_t_svm=[]
test_t_svm=[]

precision_knn=[]
recall_knn=[]
fscore_knn=[]
accuracy_knn=[]
train_t_knn=[]
test_t_knn=[]

precision_rf=[]
recall_rf=[]
fscore_rf=[]
accuracy_rf=[]
train_t_rf=[]
test_t_rf=[]

precision_mlp=[]
recall_mlp=[]
fscore_mlp=[]
accuracy_mlp=[]
train_t_mlp=[]
test_t_mlp=[]

precision_nb=[]
recall_nb=[]
fscore_nb=[]
accuracy_nb=[]
train_t_nb=[]
test_t_nb=[]

precision_dt=[]
recall_dt=[]
fscore_dt=[]
accuracy_dt=[]
train_t_dt=[]
test_t_dt=[]


X=X_res
y=y_res


for train, test in cv.split(X_res, y_res):
    pass

    
    
    rbf_svc = svm.SVC(C=50,kernel='rbf')
    t0=time()
    rbf_svc.fit(X[train], y[train])
    train_t=time()-t0
    t1=time()
    svm_predict=rbf_svc.predict(X[test])
    test_t=time()-t1
    print('accuracy using RF:',accuracy_score(svm_predict, y[test]))
    precision, recall, fscore, support = score(y[test], svm_predict)
    accuracy_svm.append(accuracy_score(svm_predict, y[test]))
    precision_svm.append(precision)
    recall_svm.append(recall)
    fscore_svm.append(fscore)
    train_t_svm.append(train_t)
    test_t_svm.append(test_t)
    
    clf = neighbors.KNeighborsClassifier(7, weights='distance')
    t0=time()
    clf.fit(X[train], y[train])
    train_t=time()-t0
    t1=time()
    knn_predict = clf.predict(X[test])
    test_t=time()-t1
    print('accuracy using RF:',accuracy_score(svm_predict, y[test]))
    precision, recall, fscore, support = score(y[test], knn_predict)
    accuracy_knn.append(accuracy_score(knn_predict, y[test]))
    precision_knn.append(precision)
    recall_knn.append(recall)
    fscore_knn.append(fscore)
    train_t_knn.append(train_t)
    test_t_knn.append(test_t)
    



    NB = GaussianNB()
    t0=time()
    NB.fit(X[train], y[train]) 
    train_t=time()-t0
    t1=time()
    preditct_NB = NB.predict(X[test])
    test_t=time()-t1
    print('accuracy using RF:',accuracy_score(preditct_NB, y[test]))
    precision, recall, fscore, support = score(y[test], preditct_NB)
    accuracy_nb.append(accuracy_score(preditct_NB, y[test]))
    precision_nb.append(precision)
    recall_nb.append(recall)
    fscore_nb.append(fscore)
    train_t_nb.append(train_t)
    test_t_nb.append(test_t)
    
    
    DT = DecisionTreeClassifier(min_samples_split=20, random_state=99)
    t0=time()
    DT.fit(X[train], y[train]) 
    train_t=time()-t0
    t1=time()
    preditct_DT = DT.predict(X[test])
    test_t=time()-t1
    print('accuracy using RF:',accuracy_score(preditct_DT, y[test]))
    precision, recall, fscore, support = score(y[test], preditct_DT)
    accuracy_dt.append(accuracy_score(preditct_DT, y[test]))
    precision_dt.append(precision)
    recall_dt.append(recall)
    fscore_dt.append(fscore)
    train_t_dt.append(train_t)
    test_t_dt.append(test_t)



    
    
    RF = RandomForestClassifier(min_samples_split=20, random_state=99,max_depth=(len(X_train)-1))
    t0=time()
    RF.fit(X[train], y[train]) 
    train_t=time()-t0
    t1=time()
    preditct_RF = RF.predict(X[test])
    test_t=time()-t1
    print('accuracy using RF:',accuracy_score(preditct_RF, y[test]))
    precision, recall, fscore, support = score(y[test], preditct_RF)
    accuracy_rf.append(accuracy_score(preditct_RF, y[test]))
    precision_rf.append(precision)
    recall_rf.append(recall)
    fscore_rf.append(fscore)
    train_t_rf.append(train_t)
    test_t_rf.append(test_t)

    
    ### MLP
    mlp_clf = MLPClassifier(solver='sgd', alpha=1e-4,hidden_layer_sizes=(10,2),learning_rate='adaptive', random_state=1,activation='tanh')
    t0=time()
    mlp_clf.fit(X[train], y[train])
    train_t=time()-t0
    t1=time()
    preditct_mlp = mlp_clf.predict(X[test])
    test_t=time()-t1
    print('accuracy using NN:',accuracy_score(preditct_mlp, y[test]))
    precision, recall, fscore, support = score(y[test], preditct_mlp)
    accuracy_mlp.append(accuracy_score(preditct_mlp, y[test]))
    precision_mlp.append(precision)
    recall_mlp.append(recall)
    fscore_mlp.append(fscore)
    train_t_mlp.append(train_t)
    test_t_mlp.append(test_t)
    
    
avg_precision_svm=np.average(precision_svm)
avg_recall_svm=np.average(recall_svm)
avg_fscore_svm=np.average(fscore_svm)
avg_accuracy_svm=np.average(accuracy_svm)
avg_train_t_svm=np.average(train_t_svm)
avg_test_t_svm=np.average(test_t_svm)


avg_precision_knn=np.average(precision_knn)
avg_recall_knn=np.average(recall_knn)
avg_fscore_knn=np.average(fscore_knn)
avg_accuracy_knn=np.average(accuracy_knn)
avg_train_t_knn=np.average(train_t_knn)
avg_test_t_knn=np.average(test_t_knn)

avg_precision_rf=np.average(precision_rf)
avg_recall_rf=np.average(recall_rf)
avg_fscore_rf=np.average(fscore_rf)
avg_accuracy_rf=np.average(accuracy_rf)
avg_train_t_rf=np.average(train_t_rf)
avg_test_t_rf=np.average(test_t_rf)

avg_precision_mlp=np.average(precision_mlp)
avg_recall_mlp=np.average(recall_mlp)
avg_fscore_mlp=np.average(fscore_mlp)
avg_accuracy_mlp=np.average(accuracy_mlp)
avg_train_t_mlp=np.average(train_t_mlp)
avg_test_t_mlp=np.average(test_t_mlp)

avg_precision_nb=np.average(precision_nb)
avg_recall_nb=np.average(recall_nb)
avg_fscore_nb=np.average(fscore_nb)
avg_accuracy_nb=np.average(accuracy_nb)
avg_train_t_nb=np.average(train_t_nb)
avg_test_t_nb=np.average(test_t_nb)

avg_precision_dt=np.average(precision_dt)
avg_recall_dt=np.average(recall_dt)
avg_fscore_dt=np.average(fscore_dt)
avg_accuracy_dt=np.average(accuracy_dt)
avg_train_t_dt=np.average(train_t_dt)
avg_test_t_dt=np.average(test_t_dt)



### feature selection - tree based
model_feat = ExtraTreesClassifier()
model_feat.fit(X, y)

model = SelectFromModel(model_feat, prefit=True)
X_new = model.transform(X)
X_new.shape
importances = model_feat.feature_importances_
std = np.std([model_feat.feature_importances_ for tree in model_feat.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.xlabel('column number')
plt.ylabel('importance')
plt.show()
plt.savefig('feature_importance.png')
