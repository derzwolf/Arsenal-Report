#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold, cross_val_score, cross_validate
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder , OrdinalEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import neighbors
from sklearn.metrics import confusion_matrix


# In[2]:


df = pd.read_csv('ENB2012_deata.csv', sep = ',')
df.head()


# In[3]:


df['Y3'] = df['Y1']+ df['Y2']


# In[4]:


q0 = df['Y3'].quantile(0)
q1 = df['Y3'].quantile(0.25)
q2 = df['Y3'].quantile(0.5)
q3 = df['Y3'].quantile(0.75)
q4 = df['Y3'].quantile(1)


# In[5]:


Y4 = pd.cut(x=df['Y3'], bins = [q0,q1, q2, q3, q4] , labels = ['0', '1', '2 ', '3'])


# In[14]:


#Y4.astype('O')


# In[12]:


#Y5 = pd.get_dummies(Y4,prefix = 'L')


# In[13]:


le = LabelEncoder()


# In[14]:


Y5 = le.fit_transform(Y4)


# In[15]:


#Y5


# In[16]:


df 


# In[17]:


df


# In[18]:


data = df.iloc[:, 0:10]
data


# In[19]:


target = Y5
target


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)


# In[21]:


sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.fit_transform(X_test)
X_test_scaled


# In[22]:


#RF
rfc = RandomForestClassifier(n_jobs=-1, random_state=321)
param_rf = [{'n_estimators': range(2,31,2),
                  'max_features': ['sqrt', 'log2', None]}]


# In[23]:


#KNN
knn = neighbors.KNeighborsClassifier(n_neighbors=1)
param_knn = {'n_neighbors':range(2, 51)}


# In[24]:


#SVM
svc = SVC(random_state=22) 
param_svc = [{'kernel': ['rbf'], 'C': [0.1,1,10,50],
           'kernel': ['linear'], 'C': [0.1,1,10,50]}]


# In[25]:


#GCV
gcv = {}
for clf, params, name in zip ((rfc, knn, svc),(param_rf,param_knn,param_svc),('RF','KNN','SVM')):
    gcv = GridSearchCV(clf, params, cv = 3, refit = True)


# In[26]:


gcv.fit(X_train_scaled, y_train)


# In[27]:


y_pred = gcv.predict(X_test_scaled)


# In[28]:


gcv.score(X_test, y_test)


# In[29]:


confusion_matrix(y_test, y_pred)


# In[30]:


gcv.best_params_


# In[31]:


gcv.best_estimator_


# In[32]:


#NCV
ncv = cross_validate(gcv,X_train_scaled, y_train)


# In[33]:


ncv


# In[ ]:





# In[ ]:





# In[ ]:




