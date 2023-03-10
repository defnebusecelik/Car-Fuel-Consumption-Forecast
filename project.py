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

ridge=Ridge(random_state=42, max_iter=10_000)
alphas= np.logspace(-4,-0.5,30)
tuned_parameters=[{"alpha":alphas}]
n_folds=5

clf=GridSearchCV(ridge,tuned_parameters,cv=n_folds,scoring="neg_mean_squared_error",refit=True)
clf.fit(X_train, Y_train)
scores=clf.cv_results_["mean_test_score"]
scores_std=clf.cv_results_["std_test_score"]

print("Ridge Coef: ",clf.best_estimator_.coef_)
#Ridge Coef:  [ 0.0523685  -0.06456481 -0.17569886 -0.00238728  0.08970587 -0.01417887
# 0.04788593 -0.01350055 -0.01809001 -0.0324085  -0.01526132  0.00769559
#0.01126857]

ridge=clf.best_estimator_
print("Ridge Best Estimator: ",ridge)
#Ridge Best Estimator:  Ridge(alpha=0.31622776601683794, max_iter=10000, random_state=42)

y_predicted_dummy=clf.predict(X_test)
mse=mean_squared_error(Y_test, y_predicted_dummy)
print("Ridge MSE: ",mse)
#Ridge MSE:  0.013126534897125423

plt.figure()
plt.semilogx(alphas,scores)
plt.xlabel("Alpha")
plt.ylabel("Score")
plt.title("Ridge")

# 9. Lasso Regression


lasso=Lasso(random_state=42, max_iter=10_000)

clf=GridSearchCV(lasso,tuned_parameters,cv=n_folds,scoring="neg_mean_squared_error",refit=True)
clf.fit(X_train, Y_train)
scores=clf.cv_results_["mean_test_score"]
scores_std=clf.cv_results_["std_test_score"]

print("Lasso Coef: ",clf.best_estimator_.coef_)
#Lasso Coef:  [-0.         -0.04694631 -0.16881732  0.          0.08523562 -0.00796941
# 0.05417585 -0.00487626  0.         -0.         -0.01223663  0.
 #0.00103414]

lasso=clf.best_estimator_
print("Lasso Best Estimator: ",lasso)
#Lasso Best Estimator:  Lasso(alpha=0.006461670787466976, max_iter=10000, random_state=42)

y_predicted_dummy=clf.predict(X_test)
mse=mean_squared_error(Y_test, y_predicted_dummy)
print("Lasso MSE: ",mse)
#Lasso MSE:  0.012996656622299515

plt.figure()
plt.semilogx(alphas,scores)
plt.xlabel("Alpha")
plt.ylabel("Score")
plt.title("Lasso")

# 10. ElasticNet

parametersGrid = {"alpha": alphas,
                  "l1_ratio": np.arange(0.0, 1.0, 0.05)}

elastic=ElasticNet(random_state=42, max_iter=10_000)

clf=GridSearchCV(elastic,parametersGrid,cv=n_folds,scoring="neg_mean_squared_error",refit=True)
clf.fit(X_train, Y_train)

print("ElasticNet Coef: ",clf.best_estimator_.coef_)
#ElasticNet Coef:  [-0.         -0.04947865 -0.16423179  0.          0.08515963 -0.00864661
#  0.05563027 -0.00563329  0.         -0.         -0.01369214  0.
#  0.00164188]

elastic=clf.best_estimator_
print("ElasticNet Best Estimator: ",elastic)
#ElasticNet Best Estimator:  ElasticNet(alpha=0.011264816923358867, max_iter=10000, random_state=42)

y_predicted_dummy=clf.predict(X_test)
mse=mean_squared_error(Y_test, y_predicted_dummy)
print("ElasticNet MSE: ",mse)
#ElasticNet MSE:  0.013084698406946649

"""
StandartScaler:
    LR Mse  0.013111602645592177
    Ridge MSE:  0.013126534897125423
    Lasso MSE:  0.012996656622299515
    ElasticNet MSE:  0.013084698406946649

"""

# 11. XGBoost

parametersGrid = {'nthread':[4], 'objective':['reg:linear'], 'learning_rate': [.03, 0.05, .07], 'max_depth': [5, 6, 7],
              'min_child_weight': [4], 'subsample': [0.7], 'colsample_bytree': [0.7],'n_estimators': [500,1000]}

model_xgb= xgb.XGBRegressor()

clf=GridSearchCV(model_xgb,parametersGrid,cv=n_folds,scoring="neg_mean_squared_error",refit=True,n_jobs=5, verbose=True)
clf.fit(X_train,Y_train)
model_xgb=clf.best_estimator_

y_predicted_dummy=clf.predict(X_test)
mse=mean_squared_error(Y_test, y_predicted_dummy)
print("XGBRegressor MSE: ",mse)
#XGBRegressor MSE:  0.014674578024248524

# 12. Averaging Models

class AveragingModels():
    def __init__(self, models):
        self.models = models
        
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        for model in self.models_:
            model.fit(X, y)
        return self
    
    def predict(self, X):
        predictions = np.column_stack([model.predict(X) for model in self.models_])
        return np.mean(predictions, axis=1)  


averaged_models = AveragingModels(models = (model_xgb, lasso))
averaged_models.fit(X_train, Y_train)

y_predicted_dummy = averaged_models.predict(X_test)
mse = mean_squared_error(Y_test,y_predicted_dummy)
print("Averaged Models MSE: ",mse)
#Averaged Models MSE:  0.012199784984546005

"""
StandartScaler:
    LR Mse  0.013111602645592177
    Ridge MSE:  0.013126534897125423
    Lasso MSE:  0.012996656622299515
    ElasticNet MSE:  0.013084698406946649
    Averaged Models MSE:  0.012199784984546005
"""
