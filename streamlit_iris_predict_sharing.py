# -*- coding: utf-8 -*-
"""chekpoint24_streamlit.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1hv1nJI6j6ntn9L7BLX7syGUDQVnczXX-
"""
import streamlit as st
from sklearn.datasets import load_iris
import numpy as  np

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


st.title("Prediction Species of iris flowers")
data_new={}

min_seplen=4.3
max_seplen=7.9
moy_seplen=5.84


data_new['sepal_length']=st.slider('Enter sepal_length(in cm)',float(min_seplen),float(max_seplen),float(moy_seplen))

min_sepwid=2
max_sepwid=4.4
moy_sepwid=3.05
data_new['sepal_width'] =st.slider('Enter sepal_width(in cm)',float(min_sepwid),float(max_sepwid),float(moy_sepwid))

min_petlen=1
max_petlen=6.9
moy_pepten=3.75
data_new['petal_length']=st.slider('Enter petal_length(in cm)',float(min_petlen),float(max_petlen),float(moy_pepten))

min_petwid=0.1
max_petwid=2.5
moy_petwid=1.19
data_new['petal_width']=st.slider('Enter petal_width(in cm)',float(min_petwid),float(max_petwid),float(moy_petwid))

data_new_df = pd.DataFrame(data_new, index=[0])

# Make prediction
if st.button('Predict species iris'):
        
        data_new_scal=std.fit_transform(data_new_df)

        pred = model.predict(data_new_scal)
        if pred[0] == 0:
            st.success('The predicted type of iris flower is  Setosa')
        elif pred[0] ==1:
        

            st.success('The predicted type of iris flower is  Versicolor')
        else:
            st.success('The predicted type of iris flower is  Virginica ')
            

# Run the web app