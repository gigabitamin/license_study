
# https://kr.object.gov-ncloudstorage.com/dataq/dataq-10th/%5BK-DATA%5D%20%EB%B9%85%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%B6%84%EC%84%9D%EA%B8%B0%EC%82%AC%20%EC%8B%A4%EA%B8%B0%20%EC%B2%B4%ED%97%98%ED%99%98%EA%B2%BD%20%EA%B0%80%EC%9D%B4%EB%93%9C_250605.pdf

# (체험) 제2유형
# 유형
# 프로그래밍
# 제공된 학습용 데이터(customer_train.csv)는 백화점 고객의 1년간 상품 구매 기록이다. 학습용 데이터를 활용하여 총구매액을 예측하는 모델을 개발하고, 이 중 가장 우수한 모델을 평가용 데이터(customer_test.csv)에 적용하여 총구매액을 예측하시오. 예측 결과는 아래의 【제출 형식】을 준수하여, CSV 파일로 생성하는 코드를 제출하시오.

#  * 예측 결과는 RMSE(Root Mean Squared Error) 평가지표에 따라 평가함

# 【제출 형식】

#   ㉠ CSV 파일명: result.csv (파일명에 디렉토리·폴더 지정불가)
#   ㉡ 예측 총 구매금액 칼럼명: pred
#   ㉢ 제출 칼럼 개수: pred 칼럼 1개 
#   ㉣ 평가용 데이터 개수와 예측 결과 데이터 개수 일치: 2,482개




# 출력을 원하실 경우 print() 함수 활용
# 예시) print(df.head())

# getcwd(), chdir() 등 작업 폴더 설정 불필요
# 파일 경로 상 내부 드라이브 경로(C: 등) 접근 불가

import pandas as pd

train = pd.read_csv("data/customer_train.csv")
test = pd.read_csv("data/customer_test.csv")

# 사용자 코딩
# print(train.info())
# print(test.info())

train_null = train.isnull().sum()
# print(train_null)
# print(train['환불금액'].isnull().sum())
# print(train[train.isnull().any(axis=0)])
# missing_info = train.isnull().sum()
# print(train.columns[train.isnull().sum() > 0])

# 데이터 전처리
# X, Y  train/test 분리
X_train = train.drop(['총구매액', '회원ID'], axis=1)
y = train['총구매액']
X_test = test.drop(['회원ID'], axis=1)
# print(X_train.shape, y.shape, X_test.shape)

# 결측치 처리
X_train['환불금액'] = X_train['환불금액'].fillna(0)
X_test['환불금액'] = X_test['환불금액'].fillna(0)
# print(X_train.isnull().sum(), '\n', X_test.isnull().sum())

# 수치형 변수 스케일링
# import sklearn.preprocessing
# print(dir(sklearn.preprocessing)) 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
num_columns = X_train.select_dtypes(exclude='object').columns
# print(num_columns)
X_train[num_columns] = scaler.fit_transform(X_train[num_columns])
X_test[num_columns] = scaler.transform(X_test[num_columns])

# 범주형 변수 인코딩
# print(set(X_test['주구매상품']) - set(X_train['주구매상품']))
# print(set(X_train['주구매상품']) - set(X_test['주구매상품']))
# print(set(X_test['주구매지점']) - set(X_train['주구매지점']))

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

# 범주형 데이터 인코딩 변환할때 fit_transfrom 과 transfrom 차이 주의, 
# transform A B D E -> 0 1 3 4, fit_transform 으로 할 경우 0 1 2 3
X_train['주구매상품'] = encoder.fit_transform(X_train['주구매상품'])
X_test['주구매상품'] = encoder.transform(X_test['주구매상품'])
X_train['주구매지점'] = encoder.fit_transform(X_train['주구매지점'])
X_test['주구매지점'] = encoder.transform(X_test['주구매지점'])

# 데이터 분리
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y, test_size=0.2, random_state=12)
# print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)

# 모델 학습 및 검증
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X_train, y_train)
y_val_pred = model.predict(X_val)

# 평가
from sklearn.metrics import root_mean_squared_error, r2_score
rmse = root_mean_squared_error(y_val, y_val_pred)
r2 = r2_score(y_val, y_val_pred)
print(rmse, r2)

# 결과저장
y_pred = model.predict(X_test)
result = pd.DataFrame(y_pred, columns=['pred'])
result.to_csv('result.csv', index=False)

# 생성 결과 확인
result = pd.read_csv('result.csv')
print(result)



# 프로세스가 시작되었습니다.(입력값을 직접 입력해 주세요)
# > 863.9588008807895 0.7089801183218358
#          pred
# 0      973.90
# 1     3406.92
# 2     3032.15
# 3      114.30
# 4       23.30
# ...       ...
# 2477   673.59
# 2478     5.00
# 2479  1875.24
# 2480   443.04
# 2481     6.61

# [2482 rows x 1 columns]

# 프로세스가 종료되었습니다.



# --------- 간략화 방법 ----


import pandas as pd

train = pd.read_csv("data/customer_train.csv")
test = pd.read_csv("data/customer_test.csv")

# 1. 데이터 유형 파악 skip .info()

# 2. 전처리
# 2-(1) X, Y, train/test set 분리
X = train.drop(['총구매액'], axis=1)
y = train['총구매액']

X_full = pd.concat([X, test], axis=0)
X_full = X_full.drop(['회원ID'], axis=1)

# print(x_full.shape)

# 2-(2) 결측치 처리
X_full['환불금액'] = X_full['환불금액'].fillna(0)

# 2-(3)수치형 변수 스케일링 skip -> 랜덤포레스트에서 스케일링이 크게 결과에 영향을 미치지 않으므로 생략

# 2-(4) 범주형 변수 인코딩 : 원핫인코딩, get_dummies : 자기가 알아서 범주형 변수인 곳만 원핫인코딩 수행
X_full = pd.get_dummies(X_full)
# print(X_full.shape)
# print(X_full)

# 3. 데이터 분리
X_train = X_full[:train.shape[0]]
X_test = X_full[train.shape[0]:]
# print(X_train.shape, X_test.shape)

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y, test_size=0.2, random_state=12)
# print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)

# 4. 모델 학습 및 검증
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X_train, y_train)
y_val_pred = model.predict(X_val)

# 5. 평가
from sklearn.metrics import root_mean_squared_error, r2_score
rmse = root_mean_squared_error(y_val, y_val_pred)
r2 = r2_score(y_val, y_val_pred)
print(rmse, r2)

y_pred = model.predict(X_test)
result = pd.DataFrame(y_pred, columns=['pred'])
result.to_csv('result.csv', index=False)

result_rd = print(pd.read_csv('result.csv'))
print(result_rd)



# 프로세스가 시작되었습니다.(입력값을 직접 입력해 주세요)
# > (2800, 72) (700, 72) (2800,) (700,)
# 890.2382795338908 0.6910066569415179
#          pred
# 0     1087.36
# 1     3360.04
# 2     3122.22
# 3      118.40
# 4       23.36
# ...       ...
# 2477   774.11
# 2478     5.00
# 2479  1905.68
# 2480   380.83
# 2481     6.18

# [2482 rows x 1 columns]
# None




# 863.9588008807895 0.7089801183218358
# 890.2382795338908 0.6910066569415179
# 축약한 것이 결과가 더 잘 나올 수도 있음



