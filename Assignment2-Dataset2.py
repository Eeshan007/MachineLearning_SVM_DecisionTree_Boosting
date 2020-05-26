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


bank_df = pd.read_csv(r"bank-additional\bank-additional-full.csv")


# In[4]:


bank_df


# In[5]:


bank_df['job'] = bank_df['job'].replace('unknown',bank_df['job'].mode()[0])
bank_df['marital'] = bank_df['marital'].replace('unknown',bank_df['marital'].mode()[0])
bank_df['education'] = bank_df['education'].replace('unknown',bank_df['education'].mode()[0])
bank_df['default'] = bank_df['default'].replace('unknown',bank_df['default'].mode()[0])
bank_df['housing'] = bank_df['housing'].replace('unknown',bank_df['housing'].mode()[0])
bank_df['loan'] = bank_df['loan'].replace('unknown',bank_df['loan'].mode()[0])
bank_df['pdays'] = bank_df['pdays'].replace(999,28)


# In[6]:


from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
bank_df["job"]=label_encoder.fit_transform(bank_df["job"])
bank_df["marital"]=label_encoder.fit_transform(bank_df["marital"])
bank_df["education"]=label_encoder.fit_transform(bank_df["education"])
bank_df["default"]=label_encoder.fit_transform(bank_df["default"])
bank_df["housing"]=label_encoder.fit_transform(bank_df["housing"])
bank_df["loan"]=label_encoder.fit_transform(bank_df["loan"])
bank_df["contact"]=label_encoder.fit_transform(bank_df["contact"])
bank_df["month"]=label_encoder.fit_transform(bank_df["month"])
bank_df["day_of_week"]=label_encoder.fit_transform(bank_df["day_of_week"])
bank_df["poutcome"]=label_encoder.fit_transform(bank_df["poutcome"])
bank_df["y"]=label_encoder.fit_transform(bank_df["y"])


# In[39]:


normalized_df = (bank_df.iloc[:,:20] - bank_df.iloc[:,:20].mean())/bank_df.iloc[:,:20].std()


# In[40]:


bank_df.iloc[:,:20] = normalized_df
bank_df


# In[41]:


X=bank_df.iloc[:,:20]
y=bank_df['y']


# In[42]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)


# In[49]:


svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[44]:


svclassifier1 = SVC(kernel='poly',gamma='auto')
svclassifier1.fit(X_train, y_train)
y_pred1 = svclassifier1.predict(X_test)
print(confusion_matrix(y_test,y_pred1))
print(classification_report(y_test,y_pred1))


# In[45]:


svclassifier2 = SVC(kernel='rbf',gamma='auto')
svclassifier2.fit(X_train, y_train)
y_pred2 = svclassifier2.predict(X_test)
print(confusion_matrix(y_test,y_pred2))
print(classification_report(y_test,y_pred2))


# In[46]:


dtclassifier = DecisionTreeClassifier()
dtclassifier.fit(X_train, y_train)
y_pred_dt = dtclassifier.predict(X_test)
print(confusion_matrix(y_test,y_pred_dt))
print(classification_report(y_test,y_pred_dt))


# In[47]:


dtclassifier1 = DecisionTreeClassifier(max_depth=7)
dtclassifier1.fit(X_train, y_train)
y_pred_dt1 = dtclassifier1.predict(X_test)
print(confusion_matrix(y_test,y_pred_dt1))
print(classification_report(y_test,y_pred_dt1))


# In[50]:


boost_classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=300, random_state=None)
boost_classifier.fit(X_train, y_train)
y_pred_boost = boost_classifier.predict(X_test)
print(confusion_matrix(y_test,y_pred_boost))
print(classification_report(y_test,y_pred_boost))


# In[51]:


boost_classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=300, random_state=None)
boost_classifier.fit(X_train, y_train)
y_pred_boost = boost_classifier.predict(X_test)
print(confusion_matrix(y_test,y_pred_boost))
print(classification_report(y_test,y_pred_boost))


# In[15]:


sv_accuracy=[]
mse=[]

cv = KFold(n_splits=5, random_state=None, shuffle=True)
for train_index, test_index in cv.split(X):
    #print("Train Index: ", train_index, "\n")
    #print("Test Index: ", test_index)

    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y.loc[train_index], y.loc[test_index]

    svclassifier = SVC(kernel='linear')
    svclassifier.fit(X_train, y_train)
    y_pred_sv = svclassifier.predict(X_test)
    sv_accuracy.append(accuracy_score(y_test,y_pred_sv))
    mse.append(mean_squared_error(y_test,y_pred_sv))

avg_mse_sv=mean(mse)
print(avg_mse_sv)
avg_accuracy_sv=mean(sv_accuracy)
print(avg_accuracy_sv)


# In[22]:


sns.lineplot(x=[1,2,3,4,5],y=sv_accuracy,color='blue',label='Accuracy')
plt.show()
sns.lineplot(x=[1,2,3,4,5],y=mse,color='red',label='MSE')
plt.show()


# In[23]:


sv_accuracy=[]
mse=[]

cv = KFold(n_splits=5, random_state=None, shuffle=False)
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


# In[24]:


sns.lineplot(x=[1,2,3,4,5],y=sv_accuracy,color='blue',label='Accuracy')
plt.show()
sns.lineplot(x=[1,2,3,4,5],y=mse,color='red',label='MSE')
plt.show()


# In[25]:


sv_accuracy=[]
mse=[]

cv = KFold(n_splits=5, random_state=None, shuffle=True)
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


# In[26]:


sns.lineplot(x=[1,2,3,4,5],y=sv_accuracy,color='blue',label='Accuracy')
plt.show()
sns.lineplot(x=[1,2,3,4,5],y=mse,color='red',label='MSE')
plt.show()


# In[ ]:





# In[28]:


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


# In[30]:


sns.lineplot(x=dt_depth,y=dt_accuracy_over_depth,color='blue',label='Accuracy')
plt.show()
sns.lineplot(x=dt_depth,y=dt_mse_over_depth,color='red',label='MSE')
plt.show()


# In[31]:


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


# In[33]:


sns.lineplot(x=dt_depth,y=dt_accuracy_over_depth,color='blue',label='Accuracy')
plt.show()
sns.lineplot(x=dt_depth,y=dt_mse_over_depth,color='red',label='MSE')
plt.show()


# In[37]:


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


# In[38]:


sns.lineplot(x=boost_depth,y=boost_accuracy_over_depth,color='blue',label='Accuracy')
plt.show()
sns.lineplot(x=boost_depth,y=boost_mse_over_depth,color='red',label='MSE')
plt.show()


# In[ ]:




