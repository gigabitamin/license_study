# License Study  
  
============================  

# 250629  
- aduino hurskey lens test : 탐색 한계, 정확도, threshold 테스트  
- 허스키 렌즈에 obj1~obj4 학습, 화면에 포착된 obj와 갯수 OLED에 연동 출력, 임계값 설정  

# 250628 
- AICE 정리, sns.load_dataset() 이용, sns.get_dataset_names() 이용해서 필요한 데이터 로딩(분류:'titanic'(생존자 분류) / 회귀:'diamonds'(가격 예측))  
- df.describe() 로 median 과 mean 이 차이나는 데이터 열 확인 (이상치 탐색, 데이터 불균형 탐색)  
- boxplot(), jointplot() 등으로 이상치 분포 확인 -> 제거 or 대체  
- df.corr(numeric_only=True) 상관관계 확인시 수치형 열만 탐색, or .select_types(include='number')  

- loss: 1342423.1250 -> loss 가 너무 크게 나올 경우, 타겟인 y도 스케일링 해준 뒤 다시 테스트
- 타겟인 가격을 스케일링 할 경우 predict 후에는 스케일링 된 예측 가격을 inverse_transform 으로 다시 원래 가격으로 변환

- valid 데이터로 predict 진행시 주의 점: X_valid는 모델이 학습할 때 직접 사용한 데이터는 아님 (학습은 X_train, y_train에서만)
- X_valid는 모델 훈련 도중 성능을 측정하기 위해 사용된 검증(validation)용 데이터이므로 오버피팅은 아니고, predict() 해서 실제 예측값과 비교하는 건 전혀 문제 없음
- 그러나 X_valid는 하이퍼파라미터 튜닝이나 조기 종료(early stopping)에 사용될 경우, 모델이 간접적으로 영향을 받았을 수 있으므로, 완전히 새로운 데이터(test set 또는 실사용 데이터)보다는 조금 더 낙관적인 결과가 나올 수 있으니 주의 필요
- 필요 시 원본 데이터인 df 를 X, y 로 분리한 후 train_test_split을 2번 중첩사용해서 train, valid, test 용으로 나눌 필요

## 250627  
- AICE 샘플 테스트 -> 빅분기 실기용 템플릿 활용  
- aduino - mblock 연동 시 milli() 함수로 비동기 구현  
- aduino pwm 4번 5번 회전 모터 : 전류(AC, DC) 입력에 따른 회전 방향, 터치패드 on/off 에 따른 회전 방향, OLED에 오실로스코프 형태 구현    
- aduino 허스키 렌즈 : 카메라에 포착된 학습 feature 갯수 FND 에 출력  

# 250626  
- ML XGBoost LightGBM -> RF 에 비해 나은 점과 적용하기 적절한 샘플 구분  
- DL callbacks 에서 Modelcheckpoint 설정 시 load_model -> best_model 로 predict 후 성능 평가 진행  
- hitory.history 에서 keys(), items() 로 loss 와 val_loss 값 확인 -> 하나의 차트에서 양 쪽 비교  

# 250625  
- AICE TEST 샘플문항 풀이 연습 : 전처리, ML, DL
- tensorflow sequantial() add.layers, Dropout, Dense
- input_shape = (X_train.shape[1]) -> 훈련 데이터의 컬럼 수와 맞춰 줌
- 예측 시(predict) y_train 과 y_pred 의 차원이 다를 경우 reshape 혹은 flatten() 으로 맞춰줌, y_train.reshape((-1, 1)), y_pred.flatten()

# 250624  
- mblock 스프라이트 아두이노 연동 pose ML 모델  
- 자동차 과속/주차 경고음, 아날로그 슬라이드 센서(0~1023)-OLED 연동, SONAR(0~1023) 센서와 FND 연동  

# 250622  
- AICE, 정처기 실기 기출 정리  
- JAVA 캡슐화, 추상화, 오버라이딩, 다형성, 상속 /  C++ Pointer, 예외처리, 메모리 자료구조, STACK, HEAP 영역  
- SCRATCH ADUINO mega 2356 블록코딩 : 사이렌 모델 - Neo, LCD, OLED, LED Matrix, RGB Led  

## 250621  
- 빅데이터 실기 모의 다시 - 합동분산 추정량, ttest_ind, f-검정, scipy.stat 이용 -> help, dir  
- train/test 컬럼이 동일하지 않을 경우 → 열 맞추기  
- x_train, x_test = x_train.align(x_test, join='left', axis=1, fill_value=0)  
- or  train 과 test 를 통합한 total 생성 후 한번에 get_dummies -> 다시 분리  

## 250620  
- 빅데이터 실기 기출변형 3~6회  
- 등분산성 levene(), 정규성 shapiro(), 독립성 검정 chi2_contingency(), 적합도 검정 chisqure()  
- 카이 검정을 위해 필요한 조치 - crosstab(a, b) index, columns 두 범주의 연관성  
- 이항분포 누적활율 scipy.stats.binom.cdf(x, n, p) - 모집단의 성공확룰 p, 표본수 n, 누적확율을 구할 값 x  
- 구간 범주화 df // 10 * 10 - 0~9, 10~19  

## 250619 
- 빅데이터분석기사 실기 대비 : 데이터구름 예제 복습 : 유형1~유형3, 회귀 유의한 or 유의하지 않은 변수 선택에 따른 상수항 추가/제외  
- 유의성을 가리기 위한 p-value 를 sklearn.ensemble 의 RandomForestRegressor 에서는 제공하지 않음  
- staticmodels.api 사용 sm.add_constant(), sm.Logit()  
- 오즈 = p / 1-p, beta 값이 x배 늘어나면 오즈비는 e**x 만큼 늘어남, np.exp() 사용 
- 이진분류가 아닐 경우 loigt 으로는 불가, 다중분류 OLS 사용  
- 로지스틱회귀 잔차 이탈도 (residual deviance) : sm.GLM 사용 model.deviance  
- 로짓 우도값 model.llf  
- 오류율 (pred_proba >= 0.5).astype('int'), 1 - acc   

## 250618  
- 빅데이터분석기사 실기 대비 : 데이터구름 예제 복습 : 유형1~유형3, 다중분류 데이터 불균형에 따른 stratify 사용여부 판단   
  
## 250617  
- 빅데이터분석기사 실기 대비 : 유형1-유형3 캐글 데이터 분석 문제 - get_dummies() X_train, X_test 컬럼수 차이 문제 해결()  
- (1) 라벨인코딩 전개 시에 불필요한 expand 전개하는 컬럼 삭제  
- (2) x_test를 x_train과 동일한 컬럼으로 재정렬하고, 없는 컬럼은 0으로 채우기  
- x_train = pd.get_dummies(x_train)  
- x_test = pd.get_dummies(x_test)  
- x_test = x_test.reindex(columns=x_train.columns, fill_value=0)  
  
## 250616  
- 빅데이터분석기사 실기 대비 : 전처리 100문제 - 이상치 처리, IQR, ANOVA 분산, 비정규성 윌콕슨  
  
## 250615  
- 빅데이터분석기사 실기 대비 : 유형3 분석 - f검정 t검정, 합동분산, p-value, ttest_ind(), 유형 1 분석 - 전처리, 결측치 제거, 4분위수 (quntaile)  
  
## 250614  
- 빅데이터분석기사 실기 대비 : 유형2 실기 문제풀이 - 데이터 전처리, 훈련, 모델 성능 테스트, 랜덤포레스트 적용  
