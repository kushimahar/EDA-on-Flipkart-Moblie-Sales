#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[41]:


file=pd.read_csv("C:\\Users\\Lenovo\\OneDrive\\Desktop\\eda_data.csv")
file


# In[42]:


file.describe()


# ###  DATA VISUALISATION :
# 
# #### INFERING THE TARGETED COLUMN SALES_PRICE WITH RESPECTIVE TO ALL OTHER FACTORS
# 
# #### Brand 

# In[43]:


file["brand"].value_counts()


# In[46]:


sns.histplot(file['brand'],kde=True)


# In[47]:


plt.figure(figsize = (25,15))
sns.pointplot(x='model',y='sales_price',hue='brand',data=file)


# #### we can observe that samsung models are reason for high sales price 

# In[48]:


plt.figure(figsize = (25,15))
sns.pointplot(x='ROM',y='sales_price',hue='brand',data=file)


# #### if the ROM is greater than 256 and the brand is samsung then there is a high chance of price being high 

# In[49]:


file["RAM"].value_counts()


# In[50]:


plt.figure(figsize = (25,15))
sns.pointplot(x='RAM',y='sales_price',hue='brand',data=file)


# #### if the RAM is greater than 8 and the model is samsung then the chance of price being high is high 

# In[51]:


file["base_color"].value_counts()


# In[52]:


plt.figure(figsize = (25,15))
sns.pointplot(x='base_color',y='sales_price',hue='brand',data=file)


# #### THE SAMSUNG BRAND HAVING THE COLOUR BRONZE IS HAVING HIGHEST PRICE 

# In[53]:


plt.figure(figsize = (25,15))
sns.pointplot(x='processor',y='sales_price',hue='brand',data=file)


# In[14]:


file["processor"].value_counts()


# #### since there are no processors named water and ciremic, there is an unwanted data, so we are going to change the processors to iOS 

# In[54]:


#map={'water':'iOS','Ceramic':'iOS','iOS':'iOS','Media Tek processor':'Media Tek processor}
file['processor'].replace({'Water':'iOS','Ceramic':'iOS'},inplace=True)
file


# In[55]:


file['processor'].unique()


# In[56]:


plt.figure(figsize = (25,15))
sns.pointplot(x='processor',y='sales_price',hue='brand',data=file)


# #### THE APPLE PROCESSOR IS HAVING THE HIGH PRICE 

# In[57]:


plt.figure(figsize = (25,15))
sns.pointplot(x='display_size',y='sales_price',hue='brand',data=file)


# #### THE SAMSUNG MODEL HAVING LARGER DISPLAY IS EFFECTING THE SALES_PRICE 

# In[59]:


plt.figure(figsize = (25,15))
sns.pointplot(x='num_rear_camera',y='sales_price',hue='brand',data=file)


# #### THE APPLE PHONE WITH TWO REAR CAMERAS EFFECTS THE SALES_PRICE 

# # INFERING THE DATA BY GROUPING IT TO: CATEGORICAL AND NUMERICAL_FEATUERS 

# In[60]:


import warnings
warnings.filterwarnings('ignore')
numerical_features=file.select_dtypes(include=np.number)
categorical_features=file.select_dtypes(include=np.object)
print("numerical_features: ", numerical_features.shape, numerical_features.columns)
print("categorical_feauters: ", categorical_features.shape, categorical_features.columns)


# In[61]:


numerical_features.iloc[:,:11].describe()
## iloc is used to coustomized set of data


# ### SKEWNESS AND KURTOSIS 

# In[62]:


pd.DataFrame({"skewness":file.skew(),"kurtosis":file.kurt()})


# #### If skewness is less than −1 or greater than +1, the distribution is highly skewed. If skewness is between −1 and −½ or between +½ and +1, the distribution is moderately skewed. If skewness is between −½ and +½, the distribution is approximately symmetric 

# In[63]:


fig, axs = plt.subplots(2,5, figsize=(20, 15), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.001)
axs = axs.ravel()

for i,j in zip([i for i in numerical_features.columns.to_list()[0:] if len(i) >0],range(10)):
    axs[j].hist(numerical_features[i])
    axs[j].set_title(i+': '+str(np.round(numerical_features[i].skew(),2)))
## ravel is used for excluding comments..


# #### ROM: left skewed RAM: it can be considered as categorical feautre display_size: right skewed
# #### num_rear_camera: it can be considered as categorical feautre
# #### num_front_camera: it can be considered as categorical feautre
# #### battery_capacity: it is highly fluctuating
# #### ratings: right skewed
# #### num_of_ratings:left skewed sales_price: left skewed discount_percent: left skewed sales: left skewed
# ###  Box-plot

# In[64]:


fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=3, ncols=2, figsize=(14,20))
AX = [ax1, ax2, ax3, ax4, ax5, ax6]
for i,j in zip([i for i in numerical_features.columns.to_list()[1:8] if len(i)>3],AX):
    sns.boxplot(x=i,y='sales_price',data=numerical_features,ax=j)


# ### CORRELATION HEAT_MAP 

# In[65]:


f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(numerical_features.corr())


# ### VARIANCE INFLATION FACTOR 

# In[66]:


vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(numerical_features.values, i) for i in range(numerical_features.shape[1])]
vif["features"] = numerical_features.columns
vif


# #### HERE WE HAVE HIGH MULTI CORELATION BETWEEN DISPLAY SIZE AND RATINGS SO WE CAN NEGLECT EITHER ANYONE OF THEM 

# ## CATEGORICAL FEATURES
# 
# #### DATA INTEGRITY AND DATA CLEANING FOR CATEGORICAL FEATURES 

# In[67]:


categorical_features.shape, categorical_features.columns


# In[68]:


for i in categorical_features.columns.to_list():
    print("Total unique values for",i,len(categorical_features[i].unique()))
    print("Value Counts for",i,'\n',categorical_features[i].value_counts(),'\n')


# In[69]:


file.head()


# In[70]:


#map={'water':'iOS','Ceramic':'iOS','iOS':'iOS','Media Tek processor':'Media Tek processor}
file['processor'].replace({'Water':'iOS','Ceramic':'iOS'},inplace=True)
from sklearn.preprocessing import LabelEncoder
cols=['brand','model','base_color','screen_size','processor']
for i in cols:
    le=LabelEncoder()
    file[i]=le.fit_transform(file[i])


# In[71]:


file.head().append(file.tail())


# In[72]:


print(file['brand'].unique())
print(file["base_color"].unique())


# #### SPLITTING AND CREATING THE TEST TRAIN MODEL 

# In[73]:


X=file.drop(['sales_price'],axis=1)
y=file['sales_price']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[74]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,y_train)


# In[75]:


import numpy
from sklearn.metrics import mean_squared_error
print('Error is : ',np.sqrt(mean_squared_error(y_test,lr.predict(X_test))))


# In[76]:


sns.distplot(y_test-lr.predict(X_test))


# In[77]:


plt.scatter(y_test,lr.predict(X_test))


# ### FINDING ERRORS MAE, MSE, RMSE 

# In[78]:


predictions = lr.predict(X_test)
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[79]:


lr.predict([[1,107,0,0,1,256,4,6.5,3,1,4250,4.7,500,0.2,40]])


# In[ ]:





# In[ ]:




