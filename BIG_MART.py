# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 09:32:16 2019

@author: Apram Sachdeva
"""

import pandas as pd
train=pd.read_csv('Train.csv')
test=pd.read_csv('Test.csv')
train['source'] = 'train'
test['source'] = 'test'
df=pd.concat([train,test])
print(df.shape,train.shape,test.shape)
df.info()

'''
Int64Index: 14204 entries, 0 to 5680
Data columns (total 12 columns):
Item_Fat_Content             14204 non-null object
Item_Identifier              14204 non-null object
Item_MRP                     14204 non-null float64
Item_Outlet_Sales            8523 non-null float64
Item_Type                    14204 non-null object
Item_Visibility              14204 non-null float64
Item_Weight                  11765 non-null float64
Outlet_Establishment_Year    14204 non-null int64
Outlet_Identifier            14204 non-null object
Outlet_Location_Type         14204 non-null object
Outlet_Size                  10188 non-null object
Outlet_Type                  14204 non-null object
dtypes: float64(4), int64(1), object(7)
memory usage: 1.4+ MB
some values are missing in item weight column and outlet size
'''

df['Item_Fat_Content'].value_counts()

'''
Low Fat    8485
Regular    4824
LF          522
reg         195
low fat     178
therefore some places LF and low fat needs to be replaced by Low Fat and reg needs to be replaced by regular
'''

df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'LF':'Low Fat','low fat':'Low Fat','reg':'Regular'})
df['Item_Fat_Content'].value_counts()

'''
Low Fat    9185
Regular    5019
'''

df['Item_Visibility'].value_counts()

#items visibility has zero values at some places so replacing them by mean of all items visibility

df['Item_Visibility']=df['Item_Visibility'].replace(0,df['Item_Visibility'].mean())
print(df[df['Item_Weight'].isnull()].shape[0])

#2439 values in item weight are missing so filluing them with mean

df['Item_Weight']=df['Item_Weight'].fillna(df['Item_Weight'].mean())
print(df[df['Outlet_Size'].isnull()].shape[0])

#4016 missing values in outlet size

df['Outlet_Size']=df['Outlet_Size'].fillna(method='bfill',inplace=False)
df['Outlet_Size'].value_counts()

#calculating outlet years

df['Outlet_Years']=2019-df['Outlet_Establishment_Year']

#calculating mean ratio for each item's visibility

visibility_avg=df.pivot_table(values='Item_Visibility', index='Item_Identifier')
df['Item_Visibility_MeanRatio'] = df.apply(lambda x: x['Item_Visibility']/visibility_avg.loc[x['Item_Identifier']],axis=1)
df['Item_Visibility_MeanRatio'].describe()

#as identifier is for indexing and would not effect sales in any way

df['Item_Category'] = df['Item_Identifier'].apply(lambda x: x[0:2])

#dropping manually converted columns using item category instead of item type and outlet years instead of outlet establishment year

df.drop(['Item_Type','Outlet_Establishment_Year'],axis=1,inplace=True)

#Encoding non numerical values as model only understands numerical values

features = ['Item_Fat_Content','Outlet_Location_Type','Item_Category','Outlet_Type','Outlet_Size']
from sklearn import preprocessing
for i in features:
    le = preprocessing.LabelEncoder()
    df[i] = le.fit_transform(df[i])

#deciding predictors and labels

predictors=['Item_Fat_Content','Item_MRP','Item_Visibility','Item_Weight','Item_Category','Item_Visibility_MeanRatio','Outlet_Location_Type','Outlet_Size','Outlet_Type','Outlet_Years']
labels=['Item_Identifier','Outlet_Identifier']
target='Item_Outlet_Sales'

#seperating train and test data

train=df.loc[df['source']=='train']
test=df.loc[df['source']=='test']


#splitting data to check the accuracy of the model

from sklearn.cross_validation import train_test_split
train1,train2=train_test_split(train,test_size=1/10,random_state=0)

# creating and training model

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(train1[predictors],train1[target])
train_pred=regressor.predict(train2[predictors])
from sklearn import metrics
score=metrics.r2_score(train2[target],train_pred)

#predicting item outlet sales for test data

test[target]=regressor.predict(test[predictors])

#saving result in a file

labels.append(target)
result=pd.DataFrame({x: test[x] for x in labels})
result.to_csv('result.csv')

#######################################################################################################################
