#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import mean


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error


# In[3]:


project_df = pd.read_csv("sgemm_product_dataset\sgemm_product.csv")


# In[4]:


project_df['Run_Avg'] = project_df.iloc[:,14:18].mean(axis=1)


# In[5]:


project_df=project_df.drop(columns=['Run1 (ms)','Run2 (ms)','Run3 (ms)','Run4 (ms)'])


# In[6]:


project_df=project_df.dropna()


# In[7]:


project_df['Run_Avg'].median()
project_df['Run_Avg'] = np.where(project_df['Run_Avg'] >= project_df['Run_Avg'].median(), 1, 0)
#Converted all the values above median to 1 and below median to zero


# In[8]:


normalized_df = (project_df.iloc[:,:14] - project_df.iloc[:,:14].mean())/project_df.iloc[:,:14].std()


# In[9]:


project_df.iloc[:,:14] = normalized_df
project_df


# In[10]:


X=project_df.iloc[:,:14]
y=project_df['Run_Avg']


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)


# In[12]:


svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[13]:


svclassifier1 = SVC(kernel='poly',gamma='auto')
svclassifier1.fit(X_train, y_train)
y_pred1 = svclassifier1.predict(X_test)
print(confusion_matrix(y_test,y_pred1))
print(classification_report(y_test,y_pred1))


# In[14]:


svclassifier2 = SVC(kernel='rbf',gamma='auto')
svclassifier2.fit(X_train, y_train)
y_pred2 = svclassifier2.predict(X_test)
print(confusion_matrix(y_test,y_pred2))
print(classification_report(y_test,y_pred2))


# In[12]:


dtclassifier = DecisionTreeClassifier()
dtclassifier.fit(X_train, y_train)
y_pred_dt = dtclassifier.predict(X_test)
print(confusion_matrix(y_test,y_pred_dt))
print(classification_report(y_test,y_pred_dt))


# In[13]:


dtclassifier1 = DecisionTreeClassifier(max_depth=7)
dtclassifier1.fit(X_train, y_train)
y_pred_dt1 = dtclassifier1.predict(X_test)
print(confusion_matrix(y_test,y_pred_dt1))
print(classification_report(y_test,y_pred_dt1))


# In[25]:


boost_classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=400, random_state=None)
boost_classifier.fit(X_train, y_train)
y_pred_boost = boost_classifier.predict(X_test)
print(confusion_matrix(y_test,y_pred_boost))
print(classification_report(y_test,y_pred_boost))


# In[ ]:


sv_accuracy=[]

cv = KFold(n_splits=5, random_state=None, shuffle=False)
for train_index, test_index in cv.split(X):
    #print("Train Index: ", train_index, "\n")
    #print("Test Index: ", test_index)

    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y.loc[train_index], y.loc[test_index]

    svclassifier = SVC(kernel='linear')
    svclassifier.fit(X_train, y_train)
    y_pred_sv = svclassifier.predict(X_test)
    sv_accuracy.append(accuracy_score(y_test,y_pred_sv))

avg_accuracy_sv=mean(sv_accuracy)
print(avg_accuracy_sv)


# In[ ]:


sns.lineplot(x=[1,2,3,4,5],y=sv_accuracy,color='blue',label='Accuracy')
plt.show()


# In[18]:


sv_accuracy=[]
mse=[]

cv = KFold(n_splits=5, random_state=None, shuffle=True)
for train_index, test_index in cv.split(X):
    #print("Train Index: ", train_index, "\n")
    #print("Test Index: ", test_index)

    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y.loc[train_index], y.loc[test_index]

    svclassifier = SVC(kernel='poly',gamma='auto')
    svclassifier.fit(X_train, y_train)
    y_pred_sv = svclassifier.predict(X_test)
    sv_accuracy.append(accuracy_score(y_test,y_pred_sv))
    mse.append(mean_squared_error(y_test,y_pred_sv))

avg_mse_sv=mean(mse)
print(avg_mse_sv)
avg_accuracy_sv=mean(sv_accuracy)
print(avg_accuracy_sv)


# In[19]:


sns.lineplot(x=[1,2,3,4,5],y=sv_accuracy,color='blue',label='Accuracy')
plt.show()
sns.lineplot(x=[1,2,3,4,5],y=mse,color='red',label='MSE')
plt.show()


# In[ ]:


sv_accuracy=[]
mse=[]

cv = KFold(n_splits=5, random_state=None, shuffle=False)
for train_index, test_index in cv.split(X):
    #print("Train Index: ", train_index, "\n")
    #print("Test Index: ", test_index)

    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y.loc[train_index], y.loc[test_index]

    svclassifier = SVC(kernel='rbf',gamma='auto')
    svclassifier.fit(X_train, y_train)
    y_pred_sv = svclassifier.predict(X_test)
    sv_accuracy.append(accuracy_score(y_test,y_pred_sv))
    mse.append(mean_squared_error(y_test,y_pred_sv))

avg_mse_sv=mean(mse)
print(avg_mse_sv)
avg_accuracy_sv=mean(sv_accuracy)
print(avg_accuracy_sv)


# In[ ]:


sns.lineplot(x=[1,2,3,4,5],y=sv_accuracy,color='blue',label='Accuracy')
plt.show()
sns.lineplot(x=[1,2,3,4,5],y=mse,color='red',label='MSE')
plt.show()


# In[ ]:


dt_accuracy=[]
dt_accuracy_over_depth=[]
dt_mse_over_depth=[]
dt_depth=[]
mse=[]

for j in range(2,20):
    cv = KFold(n_splits=5, random_state=None, shuffle=False)

    for train_index, test_index in cv.split(X):
        #print("Train Index: ", train_index, "\n")
        #print("Test Index: ", test_index)

        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]


        dtclassifier = DecisionTreeClassifier(criterion='entropy',max_depth=j)
        dtclassifier.fit(X_train, y_train)
        y_pred_dt = dtclassifier.predict(X_test)
        dt_accuracy.append(accuracy_score(y_test,y_pred_dt))
        mse.append(mean_squared_error(y_test,y_pred_dt))

    avg_accuracy_dt=mean(dt_accuracy)
    avg_mse_dt=mean(mse)
    dt_accuracy_over_depth.append(avg_accuracy_dt)
    dt_mse_over_depth.append(avg_mse_dt)
    dt_depth.append(j)


# In[ ]:


sns.lineplot(x=dt_depth,y=dt_accuracy_over_depth,color='blue',label='Accuracy')
plt.show()
sns.lineplot(x=dt_depth,y=dt_mse_over_depth,color='blue',label='MSE')
plt.show()


# In[20]:


dt_accuracy=[]
dt_accuracy_over_depth=[]
dt_mse_over_depth=[]
dt_depth=[]
mse=[]

for j in range(2,20):
    cv = KFold(n_splits=5, random_state=None, shuffle=True)

    for train_index, test_index in cv.split(X):
        #print("Train Index: ", train_index, "\n")
        #print("Test Index: ", test_index)

        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]


        dtclassifier = DecisionTreeClassifier(criterion='gini',max_depth=j)
        dtclassifier.fit(X_train, y_train)
        y_pred_dt = dtclassifier.predict(X_test)
        dt_accuracy.append(accuracy_score(y_test,y_pred_dt))
        mse.append(mean_squared_error(y_test,y_pred_dt))

    avg_accuracy_dt=mean(dt_accuracy)
    avg_mse_dt=mean(mse)
    dt_accuracy_over_depth.append(avg_accuracy_dt)
    dt_mse_over_depth.append(avg_mse_dt)
    dt_depth.append(j)


# In[22]:


sns.lineplot(x=dt_depth,y=dt_accuracy_over_depth,color='blue',label='Accuracy')
plt.show()
sns.lineplot(x=dt_depth,y=dt_mse_over_depth,color='red',label='MSE')
plt.show()


# In[23]:


boost_accuracy=[]
boost_accuracy_over_depth=[]
boost_mse_over_depth=[]
boost_depth=[]
mse=[]

for i in range(2,6):
    cv = KFold(n_splits=5, random_state=None, shuffle=True)

    for train_index, test_index in cv.split(X):
        #print("Train Index: ", train_index, "\n")
        #print("Test Index: ", test_index)

        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]

        boost_classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=i), n_estimators=300, random_state=None)
        boost_classifier.fit(X_train, y_train)
        y_pred_boost = boost_classifier.predict(X_test)
        boost_accuracy.append(accuracy_score(y_test,y_pred_boost))
        mse.append(mean_squared_error(y_test,y_pred_boost))

    avg_mse_boost=mean(mse)
    avg_accuracy_boost=mean(boost_accuracy)
    boost_mse_over_depth.append(avg_mse_boost)
    boost_accuracy_over_depth.append(avg_accuracy_boost)
    boost_depth.append(i)


# In[24]:


sns.lineplot(x=boost_depth,y=boost_accuracy_over_depth,color='blue',label='Accuracy')
plt.show()
sns.lineplot(x=boost_depth,y=boost_mse_over_depth,color='red',label='MSE')
plt.show()


# In[ ]:





# In[35]:


dt_accuracy=[]
dt_accuracy_over_folds=[]
dt_folds=[]
boost_accuracy=[]
boost_accuracy_over_folds=[]
boost_folds=[]

for i in range(4,8):
    cv = KFold(n_splits=i, random_state=None, shuffle=True)

    for train_index, test_index in cv.split(X):
        #print("Train Index: ", train_index, "\n")
        #print("Test Index: ", test_index)

        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]

        dtclassifier = DecisionTreeClassifier()
        dtclassifier.fit(X_train, y_train)
        y_pred_dt = dtclassifier.predict(X_test)
        dt_accuracy.append(accuracy_score(y_test,y_pred_dt))
        
        boost_classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=300, random_state=None)
        boost_classifier.fit(X_train, y_train)
        y_pred_boost = boost_classifier.predict(X_test)
        boost_accuracy.append(accuracy_score(y_test,y_pred_boost))
        
    avg_accuracy=mean(dt_accuracy)
    dt_accuracy_over_folds.append(avg_accuracy)
    dt_folds.append(i)
    
    avg_accuracy_boost=mean(boost_accuracy)
    boost_accuracy_over_folds.append(avg_accuracy_boost)
    boost_folds.append(i)


# In[36]:


sns.lineplot(x=dt_folds,y=dt_accuracy_over_folds,label='dt')
sns.lineplot(x=boost_folds,y=boost_accuracy_over_folds,label='boost')
plt.show()


# In[ ]:




