
import streamlit as st
from sklearn.datasets import load_iris
import numpy as  np
import  pickle

iris = load_iris()
X = iris.data
y = iris.target




import pandas as pd
df=pd.DataFrame(X,columns=iris.feature_names)


df.isnull().sum()

#standarisation
from sklearn.preprocessing import StandardScaler
std = StandardScaler()
X = std.fit_transform(X)


from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
model=RandomForestClassifier(n_estimators=10)
model.fit(x_train, y_train)  #Training our model

y_t_pred=model.predict(x_test)  #testing our model
#y_t_pred

#metrique de performance data test
#from matplotlib.projections.polar import math
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score

from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score
precision_t=precision_score(y_test,y_t_pred, average="weighted")
recall_score_t=recall_score(y_test,y_t_pred,average="weighted")
accurancy_t=accuracy_score(y_test,y_t_pred)
print(precision_t)
print(recall_score_t)
print(accurancy_t)
cl_rpport=classification_report(y_test,y_t_pred)
print(cl_rpport)

#metrique de performance data training
#from matplotlib.projections.polar import math
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score
y_tr_pred=model.predict(x_train)  #testing our model
#y_tr_pred
precision_tr=precision_score(y_train,y_tr_pred,average="weighted")
recall_score_tr=recall_score(y_train,y_tr_pred,average="weighted")
accurancy_tr=accuracy_score(y_train,y_tr_pred)
print(precision_tr)
print(recall_score_tr)
print(accurancy_tr)
cl_rpport=classification_report(y_train,y_tr_pred)
print(cl_rpport)

# save the model to disk
filename = 'iris_model_randomForest.sav'
pickle.dump(model, open(filename, 'wb'))
