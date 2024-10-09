#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import immporant lobraries 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[13]:


# laod the data set 
df_mobile=pd.read_csv(r'C:\Users\pc\Desktop\Processed_Flipdata.csv')


# In[72]:


# check firt three row of data set
df_mobile.head(3)


# In[15]:


# drop unnecessary column
df_mobile=df_mobile.drop(['Unnamed: 0'], axis=1)


# In[16]:


# Check for general infirmation of data Like (count, mena, standard daviation etc)
df_mobile.describe()


# In[17]:


df_mobiel=df_mobile.rename(index={'Prize':'Price'})
df_mobile.head(3)


# In[18]:


# Number of Row and column
df_mobile.shape


# In[19]:


# Checking for data type
df_mobile.info()


# In[20]:


df_mobile.describe()


# In[21]:


# Null value in data set
df_mobile.isnull().sum()


# In[22]:


# Bifercate culumn into catagorical and numrical column 
catagorial_col=df_mobile.select_dtypes(include=['object'])
numerica_col=df_mobile.select_dtypes(include=['float','int'])
catagorial_col


# In[ ]:





# In[23]:


#numerical column
numerica_col


# In[24]:


# heat map for checking the reration between culumn 
sns.heatmap(numerica_col.corr(), annot=True)


# In[25]:


# encode catagorical data into numrical 
from sklearn.preprocessing import LabelEncoder
cols=['Model','Colour','Rear Camera','Front Camera','Processor_','Prize']
df_mobile[cols]=df_mobile[cols].apply(LabelEncoder().fit_transform)
df_mobile.head(3)


# In[26]:


#corelation beteen data 
df_mobile.corr()


# In[27]:


sns.heatmap(df_mobile.corr(),annot=True)


# In[28]:


# heat map for null value 
sns.heatmap(df_mobile.isnull(),yticklabels=False, cbar=False)


# In[29]:


# Count plot for ram distribution 
plt.figure(figsize=(8,6),)
sns.countplot(x='RAM',data=df_mobile)
plt.title('Count Of RAM')


# In[30]:


# Bar plot to show relation between RAM and Price
plt.figure(figsize=(20,10))
plt.title('RAM vs Price')
sns.barplot(x=df_mobile['RAM'],y=df_mobile['Prize'])


# In[31]:


# Count plot of Memory 

plt.figure(figsize=(8,6),)
sns.countplot(x='Memory',data=df_mobile)
plt.title('Memory')


# In[32]:


# box Plot to shwo relation between memory and price 
plt.figure(figsize=(20,10))
plt.title('Memory vs Price')
sns.barplot(x=df_mobile['Memory'],y=df_mobile['Prize'])


# In[33]:


# Count Plot of mobiel hight 
plt.figure(figsize=(8,6),)
sns.countplot(x='Mobile Height',data=df_mobile)
plt.title('Count Of Mobile Hight')


# In[34]:


# Graph to show relation between Mobile Heigh and price.  
plt.figure(figsize=(20,10))
plt.title('Mobile Height vs Price')
sns.barplot(x=df_mobile['Mobile Height'],y=df_mobile['Prize'])


# In[35]:


# box plot for checking outlier 
sns.boxplot(y=df_mobile['Prize'])


# In[36]:


data=df_mobile.values
x=data[:,0:10]
y=data[:,10]


# In[37]:


# feture selection filter method 
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[38]:


x=df_mobile.iloc[:,0:10] # independent columsn 
y=df_mobile.iloc[:,[-1]]  # dependet columns 
x,y


# In[39]:


from sklearn.ensemble import ExtraTreesClassifier
model= ExtraTreesClassifier()
model.fit(x,y)


# In[40]:


important_features=pd.Series(model.feature_importances_,index=x.columns)
important_features.nlargest(10).plot(kind='bar')
plt.show()


# In[41]:


# random forest regression 
X = df_mobile.drop(['Prize'], axis=1)

y = df_mobile['Prize']


# In[42]:


# split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)


# In[43]:


X_train.shape, X_test.shape


# In[44]:


# import library 
from sklearn.ensemble import RandomForestRegressor
rf_regresor = RandomForestRegressor(n_estimators=100, random_state=42)


# In[47]:


# Train the classifier
rf_regresor.fit(X_train,y_train)


# In[49]:


# Make predictions on the test set
y_pred = rf_regresor.predict(X_test)
y_pred


# In[57]:


# check for accuracy 
from sklearn.metrics import accuracy_score
_accuracy=accuracy_score(y_pred,y_test)


# In[54]:


#meand squard error and r2 test 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mse, r2


# In[55]:


# logistic regrasion 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc


# In[56]:


clf = LogisticRegression(random_state=0)
clf.fit(X_train, y_train)


# In[59]:


# Prediction
y_pred = clf.predict(X_test)
y_pred


# In[61]:


# accuracy
accuracu_ln= accuracy_score(y_test, y_pred)
accuracu_ln


# In[62]:


#meand squard error and r2 test
mse_ln = mean_squared_error(y_test, y_pred)
r2_ln = r2_score(y_test, y_pred)
mse_ln, r2_ln


# In[63]:


# suport vecot regression 
from sklearn.svm import SVR


# In[65]:


svr = SVR(kernel='linear')
svr


# In[66]:


# train the model on the data
svr.fit(X, y)


# In[69]:


# make predictions on the data
y_pred_svr = svr.predict(X)
y_pred_svr


# In[70]:


# accuracy 
accuracu_svr= accuracy_score(y_test, y_pred)
accuracu_svr


# In[71]:


#meand squard error and r2 test
mse_svr = mean_squared_error(y_test, y_pred)
r2_svr=r2_score(y_test, y_pred)
mse_svr, r2_svr


# In[ ]:




