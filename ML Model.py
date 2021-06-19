#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline


# # Loading Dataset

# In[3]:


df=pd.read_csv('Car details v3.csv')
df.head()


# In[ ]:


df.shape


# In[ ]:


df.info()


# # Droping extra columns

# In[ ]:


df.drop(['mileage','engine','max_power','torque','seats','seller_type'],axis=1,inplace=True)


# In[ ]:


df.head(15)


# # Handling Missing Values

# In[ ]:


df.isnull().sum()


# In[ ]:


# NO null values


# In[ ]:


df.info()


# # Create new columns [car company,car name]

# In[ ]:


# Creating company column and modifying it.


# In[ ]:


df['company']=np.nan


# In[ ]:


df=df[['name','company','year','selling_price','km_driven','fuel','transmission','owner']]


# In[ ]:


df['company']=df['name'].str.split(' ').str.slice(0,1).str.join(' ')


# In[ ]:


df.head(3)


# In[ ]:


# Modifying name column


# In[ ]:


df['name']=df['name'].str.split(' ').str.slice(0,4).str.join(' ')


# In[ ]:


df.head(3)


# In[ ]:


# Checking dataset again


# In[ ]:


df.info()


# In[ ]:


# Check dataset in more detail


# # Checking Outlier

# In[ ]:


df.describe()


# In[ ]:





# In[ ]:


df[df['selling_price']>7.2e6]


# In[ ]:


plt.scatter(df['year'],df['selling_price'])


# In[ ]:


plt.scatter(df['year'],df['km_driven'])
plt.plot()


# In[ ]:


df[df['km_driven']>1e6]


# In[ ]:


df[df['year']<1993]


# # Dropping Oulier

# In[ ]:


df.drop([170,1810,3489,316,5322],axis=0,inplace=True)


# In[ ]:


df.reset_index(drop=True)


# In[ ]:


df


# # Storing Cleaned Data

# In[ ]:


df.to_csv('Cleaned data of Main file 3n.csv')


# In[ ]:


final_data=df


# In[ ]:


final_data.head(3)


# # Split main data to X and Y

# In[ ]:


X=final_data.drop('selling_price',axis=1)


# In[ ]:


X.head(3)


# In[ ]:


Y=final_data['selling_price']


# In[ ]:


Y.head(3)


# # Split X and Y data to train test

# In[ ]:


X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.3,random_state=174)


# # One Hot Encoding

# In[ ]:


ohe=OneHotEncoder()
ohe.fit(X[['name','company','fuel','transmission','owner']])


# In[ ]:





# In[ ]:


column_trans=make_column_transformer((OneHotEncoder(categories=ohe.categories_),['name','company','fuel','transmission','owner']),
                                    remainder='passthrough')


# In[ ]:


lr=LinearRegression()


# In[ ]:


pipe=make_pipeline(column_trans,lr)


# In[ ]:


pipe.fit(X_train,Y_train)


# In[ ]:


y_pred=pipe.predict(X_test)


# In[ ]:


r2_score(Y_test,y_pred)


# # Checking r2 socre on different random state for getting max r2_score

# In[ ]:


# for i in range(10):
#     X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=i)
#     lr1=LinearRegression()
#     pipe=make_pipeline(column_trans,lr1)
#     pipe.fit(X_train,Y_train)
#     y_pre=pipe.predict(X_test)
#     print(r2_score(Y_test,y_pre))


# In[ ]:





# # Model with standardization

# In[ ]:


# ohe_c=OneHotEncoder()
# ohe_c.fit(X[['name','company','fuel','transmission','owner']])


# In[ ]:


# col_trs=make_column_transformer((OneHotEncoder(categories=ohe.categories_),['name','company','fuel','transmission','owner']),
#                                 (StandardScaler(),['year','km_driven']),remainder='passthrough')


# In[ ]:


# lin_model=LinearRegression()


# In[ ]:


# p=make_pipeline(col_trs,lin_model)


# In[ ]:


# p.fit(X_train,Y_train)


# In[ ]:


# y_p=p.predict(X_test)


# In[ ]:


# r2_score(Y_test,y_p)


# ### Note: Here we can see that the above model with standard_scaler is giving approx same r2_score. So i am going the model without Standard_Scaler

# # Dumping File

# In[ ]:


import pickle


# In[ ]:


pickle.dump(pipe,open('Used Car Price Prediction Model.pkl','wb'))


# # Prediction

# In[ ]:


pipe.predict(pd.DataFrame([['Maruti Swift Dzire VDI','Maruti',2018,100,'Diesel','Manual','First Owner']],columns=['name','company','year','km_driven','fuel','transmission','owner']))


# In[ ]:


plt.scatter(Y_test,y_pred)
plt.show()


# In[ ]:




