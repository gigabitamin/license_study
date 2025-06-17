


# 유형 2
## 서비스 이탈예측 데이터

# 데이터 설명 : 고객의 신상정보 데이터를 통한 회사 서비스 이탈 예측 (종속변수 : Exited)  
# x_train : https://raw.githubusercontent.com/Datamanim/datarepo/main/churnk/X_train.csv  
# y_train : https://raw.githubusercontent.com/Datamanim/datarepo/main/churnk/y_train.csv  
#x_test : https://raw.githubusercontent.com/Datamanim/datarepo/main/churnk/X_test.csv  
# x_label(평가용) : https://raw.githubusercontent.com/Datamanim/datarepo/main/churnk/y_test.csv  
# 데이터 출처 : https://www.kaggle.com/shubh0799/churn-modelling 에서 변형  


import pandas as pd
#데이터 로드
x_train = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/churnk/X_train.csv")
y_train = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/churnk/y_train.csv")
x_test= pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/churnk/X_test.csv")

print(x_train.head())
print(y_train.head())

# x_train = x_train.drop('CustomerId', axis=1)
# x_train.shape
# x_test = x_test.drop('CustomerId', axis=1)
# x_test.shape
# y_train = y_train.drop('CustomerId', axis=1)
# y_train.shape


x_train.isnull().sum()
x_test.isnull().sum()
y_train.isnull().sum()

x_train.shape
x_test.shape
y_train.shape

# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier

# x_train = pd.get_dummies(x_train)
# x_test = pd.get_dummies(x_test)
# x_test = x_test[x_train.columns] # 컬럼수 차이로 인한 에러 발생 2305 vs 1704




drop_col = ['CustomerId','Surname']
x_train_drop = x_train.drop(columns = drop_col)
x_test_drop = x_test.drop(columns = drop_col)

# import sklearn
# print(sklearn.__all__)
# import sklearn.model_selection
# print(dir(sklearn.model_selection))

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
x_train_dummies = pd.get_dummies(x_train_drop)
y = y_train['Exited']


x_test_dummies = pd.get_dummies(x_test_drop)
# train과 컬럼 순서 동일하게 하기 (더미화 하면서 순서대로 정렬을 이미 하기 때문에 오류가 난다면 해당 컬럼이 누락된것)
x_test_dummies = x_test_dummies[x_train_dummies.columns]
# print(help(train_test_split))



