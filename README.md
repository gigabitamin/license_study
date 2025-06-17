# license_study

============================

## 250614
빅데이터분석기사 실기 대비 : 유형2 실기 문제풀이 - 데이터 전처리, 훈련, 모델 성능 테스트, 랜덤포레스트 적용

## 250615
빅데이터분석기사 실기 대비 : 유형3 분석 - f검정 t검정, 합동분산, p-value, ttest_ind(), 유형 1 분석 - 전처리, 결측치 제거, 4분위수 (quntaile)

## 250616
빅데이터분석기사 실기 대비 : 전처리 100문제 - 이상치 처리, IQR, ANOVA 분산, 비정규성 윌콕슨

## 250617
빅데이터분석기사 실기 대비 : 유형1-유형3 캐글 데이터 분석 문제 - get_dummies() X_train, X_test 컬럼수 차이 문제 해결()
(1) 라벨인코딩 전개 시에 불필요한 expand 전개하는 컬럼 삭제
(2) x_test를 x_train과 동일한 컬럼으로 재정렬하고, 없는 컬럼은 0으로 채우기
x_train = pd.get_dummies(x_train)
x_test = pd.get_dummies(x_test)
x_test = x_test.reindex(columns=x_train.columns, fill_value=0)

## 250618
빅데이터분석기사 실기 대비 : 데이터구름 예제 복습 : 유형1~유형3

