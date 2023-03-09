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
data=data[filter_horse] #397 #remove horsepower outlier


acceleration_desc=describe["Acceleration"]
q3_acce=acceleration_desc[6]
q1_acce=acceleration_desc[4]
IQR_acce=q3_acce-q1_acce
toplimit_acce=q3_acce+(throw*IQR_acce)
botlimit_acce=q1_acce-(throw*IQR_acce)

filter_acce_bot= botlimit_acce < data["Acceleration"]
filter_acce_top= toplimit_acce > data["Acceleration"]
filter_acce= filter_acce_bot & filter_acce_top
data=data[filter_acce] #395 #remove acceleration outlier


# 5. Feature Engineering

# 5.1 Skewness

sns.distplot(data.Target, fit = norm)

(mu, sigma) = norm.fit(data["Target"])
print("mu: {}, sigma = {}".format(mu, sigma))
#mu: 23.472405063291134, sigma = 7.756119546409932
plt.figure()
stats.probplot(data["Target"], plot=plt)
plt.show()

data["Target"]=np.log1p(data["Target"])
plt.figure()
sns.distplot(data.Target, fit=norm)

(mu, sigma) = norm.fit(data["Target"])
print("mu: {}, sigma = {}".format(mu, sigma))
#mu: 3.146474056830183, sigma = 0.3227569103044823
plt.figure()
stats.probplot(data["Target"], plot=plt)
plt.show()

skewed_features= data.apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness=pd.DataFrame(skewed_features, columns=["Skewed"])

# 5.2 One Hot Encoding
#categorical variables
data["Cylinders"]=data["Cylinders"].astype(str)
data["Origin"] = data["Origin"].astype(str)
data=pd.get_dummies(data)

#6. Split & Standardization

x=data.drop(["Target"],axis=1)
y=data.Target

X_train, X_test, Y_train, Y_test= train_test_split(x,y,test_size=0.7,random_state=42)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

#7. Linear Regression

lr=LinearRegression()
lr.fit(X_train,Y_train)
print("LR Coef: ",lr.coef_)
#LR Coef:  [ 0.05946345 -0.06548775 -0.18011713 -0.00191276  0.09021617 -0.01364615
 # 0.04858425 -0.01330318 -0.01824245 -0.03324265 -0.01547399  0.00799193
 # 0.0112631 ]
y_predicted_dummy=lr.predict(X_test)
mse=mean_squared_error(Y_test,y_predicted_dummy)
print("LR Mse ",mse)
#LR Mse  0.013111602645592177

#8. Ridge Regression


