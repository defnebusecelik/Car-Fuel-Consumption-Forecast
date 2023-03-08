# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 22:55:44 2023

@author: daphn
"""

# 1.Import Library
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm,skew
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.base import clone
import xgboost as xgb

#for warnings
import warnings 
warnings.filterwarnings("ignore")


# 2. Import Data/ Problem Description
column_name = ["MPG", "Cylinders", "Displacement","Horsepower","Weight","Acceleration","Model Year", "Origin"]
data = pd.read_csv("data.data", names = column_name, na_values = "?", comment = "\t",sep = " ", skipinitialspace = True)

data = data.rename(columns = {"MPG":"Target"})

data.info()
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 398 entries, 0 to 397
Data columns (total 8 columns):
 #   Column        Non-Null Count  Dtype  
---  ------        --------------  -----  
 0   Target        398 non-null    float64
 1   Cylinders     398 non-null    int64  
 2   Displacement  398 non-null    float64
 3   Horsepower    392 non-null    float64
 4   Weight        398 non-null    float64
 5   Acceleration  398 non-null    float64
 6   Model Year    398 non-null    int64  
 7   Origin        398 non-null    int64  
dtypes: float64(5), int64(3)
memory usage: 25.0 KB """

describe= data.describe()
# 6 missing value -->Horsepower

# 3. Missing value
print(data.isna().sum()) #6
data["Horsepower"]=data["Horsepower"].fillna(data["Horsepower"].mean())
print(data.isna().sum()) #0
sns.distplot(data.Horsepower)


# 4. EDA ( Exploratory Data Analysis)
correlation=data.corr()
sns.clustermap(correlation, annot=True, fmt=".3f")
plt.title("Correlation map")
plt.show()

threshold=0.75
filtre=np.abs(correlation["Target"])>threshold
correlation_features=correlation.columns[filtre].tolist()
sns.clustermap(data[correlation_features].corr(), annot=True, fmt=".3f")
plt.title("Correlation map 2")
plt.show()

sns.pairplot(data, diag_kind = "kde", markers = "+")
plt.show()

# feature engineering
plt.figure()
sns.countplot(data["Cylinders"])
print(data["Cylinders"].value_counts())
"""
4    204
8    103
6     84
3      4
5      3
Name: Cylinders, dtype: int64
"""

plt.figure()
sns.countplot(data["Origin"])
print(data["Origin"].value_counts())
"""
1    249
3     79
2     70
Name: Origin, dtype: int64
"""

for i in data.columns:
    plt.figure()
    sns.boxplot(x=i, data=data, orient="h")
    
    
# 4. Outlier

horsepower_desc=describe["Horsepower"]
q3_horse=horsepower_desc[6]
q1_horse=horsepower_desc[4]
IQR_horse=q3_horse-q1_horse
throw=2
toplimit_horse=q3_horse+(throw*IQR_horse)
botlimit_horse=q1_horse-(throw*IQR_horse)

filter_horse_bot=botlimit_horse < data["Horsepower"]
filter_horse_top=toplimit_horse > data["Horsepower"]
filter_horse= filter_horse_bot & filter_horse_top
data=data[filter_horse]

