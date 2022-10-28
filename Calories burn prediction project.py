#!/usr/bin/env python
# coding: utf-8

# This is calories burn project  made by using 'supervised Regression' by Dheeraj Yadav for submission on coursera IBM Professional certificate in machine learning.In this project XGBoost Regressor model is used, data is taken from kaggle.
# 

# In[1]:


import import_ipynb
import xgboost


# # Importing library
# 
# 

# In[2]:


import numpy as np
import pandas as ps
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost import XGBRegressor


# # data collecting
# 

# In[3]:


calories = ps.read_csv('Downloads/calories.csv')


# In[4]:


#for checking dataframe of calories.csv first five datasets
calories.head()


# In[5]:


exrcdata = ps.read_csv('Downloads/exercise.csv')


# In[6]:


exrcdata.head


# In[7]:


calories_data = ps.concat([exrcdata, calories['Calories']], axis=1)


# In[8]:


calories_data.head


# In[9]:


#checking info about calories_data


# In[10]:


calories_data.info()


# In[11]:


#checking missing number
calories_data.isnull().sum()


# In[ ]:


#statical measurement of calories_data
calories_data.describe()


# Data visualization

# In[ ]:


sns.set()


# In[ ]:


#plotting a gender column in count plot
sns.countplot(calories_data['Gender'])


# In[ ]:


#distribution of age plot
sns.distplot(calories_data['Age'])


# In[ ]:


#distribution of weight
sns.distplot(calories_data['Weight'])


# In[ ]:


#distribution of Height plot
sns.distplot(calories_data['Height'])


# In[19]:


#distribution of Heart_rate
sns.distplot(calories_data['Heart_Rate'])


# In[20]:


#distribution of Duration
sns.distplot(calories_data['Duration'])


# # finding correlation in dataset

# In[21]:


#positive correlation directly proportional 
#negative correlattion inversely proportional to calories in this particular dataset


# In[22]:


correlation = calories_data.corr()


# In[23]:


#plotting a heatmap to understand correlation
plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True , fmt='.1f', annot=True , annot_kws={'size':8}, cmap='Blues')


# # converting text value in numeric value of gender

# In[24]:


calories_data.replace({"Gender":{'male:0', 'female:1'}}, inplace=True)


# In[25]:


calories_data.head()


# In[26]:


X = calories_data.drop(columns=['User_ID' , 'Calories'], axis=1)
Y= calories_data['Calories']


# In[27]:


print(X)


# In[28]:


print(Y)


# # Now splitting data into training data and test data

# In[47]:


X_train , X_test , Y_train, Y_test= train_test_split(X,Y , test_size=0.2, random_state=2)


# In[30]:


print(X.shape, X_test.shape, X_train.shape)


# # Model training ... xgboost , XGBRegressor

# In[42]:


#loading model
model = XGBRegressor()


# In[43]:


#training model with X_train

model.fit(X_train,Y_train)


# In[77]:


##evaluation


# In[44]:


#predict the model
test_data_prediction=model.predict(X_test)


# In[45]:


print(test_data_prediction)


# # mean absolute error
# 

# In[46]:


mae=metrics.mean_absolute_error(Y_test,test_data_prediction)


# In[ ]:




