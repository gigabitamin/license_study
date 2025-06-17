# 이직여부 판단 데이터
# Attention

# 데이터 설명 : 이직여부 판단 데이터 (target: 1: 이직 , 0 : 이직 x)
# x_train: https://raw.githubusercontent.com/Datamanim/datarepo/main/HRdata/X_train.csv
# y_train: https://raw.githubusercontent.com/Datamanim/datarepo/main/HRdata/y_train.csv
# x_test: https://raw.githubusercontent.com/Datamanim/datarepo/main/HRdata/X_test.csv
# x_label(평가용) : https://raw.githubusercontent.com/Datamanim/datarepo/main/HRdata/y_test.csv
# 데이터 출처 :https://www.kaggle.com/datasets/arashnic/hr-analytics-job-change-of-data-scientists (참고, 데이터 수정)

import pandas as pd
#데이터 로드
x_train = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/HRdata/X_train.csv")
y_train = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/HRdata/y_train.csv")
x_test= pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/HRdata/X_test.csv")


print(x_train.head())
print(y_train.head())


# print(x_train.info())
# print(x_train.nunique())
# print(x_train.isnull().sum())
# 결측치가 있지만 따로 처리하지 않고 더미화


# 범주형 변수인데 적당히 많은 unique값을 가진 컬럼은 날린다.
drop_col = ['enrollee_id','city','company_type','experience']
x_train_drop = x_train.drop(columns = drop_col)
x_test_drop = x_test.drop(columns = drop_col)

# import sklearn
# print(sklearn.__all__)
# import sklearn.model_selection
# print(dir(sklearn.model_selection))

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
x_train_dummies = pd.get_dummies(x_train_drop)
y = y_train['target'].astype('int')


x_test_dummies = pd.get_dummies(x_test_drop)
# train과 컬럼 순서 동일하게 하기 (더미화 하면서 순서대로 정렬을 이미 하기 때문에 오류가 난다면 해당 컬럼이 누락된것)
x_test_dummies = x_test_dummies[x_train_dummies.columns]
# print(help(train_test_split))



X_train, X_validation, Y_train, Y_validation = train_test_split(x_train_dummies, y, test_size=0.33, random_state=42)
rf = RandomForestClassifier(random_state =23)
rf.fit(X_train,Y_train)

# import sklearn.metrics
# print(dir(sklearn.metrics))

from sklearn.metrics import accuracy_score , f1_score, recall_score, roc_auc_score ,precision_score

#model_score
predict_train_label = rf.predict(X_train)
predict_train_proba = rf.predict_proba(X_train)[:,1]

predict_validation_label = rf.predict(X_validation)
predict_validation_prob = rf.predict_proba(X_validation)[:,1]


# 문제에서 묻는 것에 따라 모델 성능 확인하기
# 정확도 (accuracy) , f1_score , recall , precision -> model.predict로 결과뽑기
# auc , 확률이라는 표현있으면 model.predict_proba로 결과뽑고 첫번째 행의 값을 가져오기 model.predict_proba()[:,1]
print('train accuracy :', accuracy_score(Y_train,predict_train_label))
print('validation accuracy :', accuracy_score(Y_validation,predict_validation_label))
print('\n')
print('train f1_score :', f1_score(Y_train,predict_train_label))
print('validation f1_score :', f1_score(Y_validation,predict_validation_label))
print('\n')
print('train recall_score :', recall_score(Y_train,predict_train_label))
print('validation recall_score :', recall_score(Y_validation,predict_validation_label))
print('\n')
print('train precision_score :', precision_score(Y_train,predict_train_label))
print('validation precision_score :', precision_score(Y_validation,predict_validation_label))
print('\n')
print('train auc :', roc_auc_score(Y_train,predict_train_proba))
print('validation auc :', roc_auc_score(Y_validation,predict_validation_prob))


# test데이터 마찬가지 위와 같은 방식
predict_test_label = rf.predict(x_test_dummies)
predict_test_proba = rf.predict_proba(x_test_dummies)[:,1]


# accuracy, f1_score, recall, precision 
#pd.DataFrame({'enrollee_id': x_test.enrollee_id, 'target': predict_test_label}).to_csv('003000000.csv', index=False)

# auc, 확률
#pd.DataFrame({'enrollee_id': x_test.enrollee_id, 'target': predict_test_proba}).to_csv('003000000.csv', index=False)

