# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 11:00:06 2019

@author: HP
"""
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
df=pd.read_csv('C:/Users/HP/Desktop/breast cancer.csv')
df1=df['area_mean']
df2=df['area_se']
df3=df['area_worst']
df['Area_mean']=df1
df['Area_se']=df2
df['Area_worst']=df3
X=df.iloc[:,33:]
#print(X)
Y=df['diagnosis']
#df['area_mean']=preprocessing.scale(df['area_mean'])
#area=(df['area_mean']-df['area_se']-df['area_worst'])
#smoothness=(df['smoothness_mean']-df['smoothness_se']-df['smoothness_worst'])
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#X=np.array(area,smoothness).reshape(-1,1)
#Y=df['diagnosis']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=44)
clf = LogisticRegression(penalty='l1',C = 1,max_iter=100)
clf.fit(X_train,Y_train)
accur=clf.score(X_test,Y_test)*100
print(accur)