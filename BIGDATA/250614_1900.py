import pandas as pd

train = pd.read_csv("data/customer_trian.csv")
test = pd.read_csv("data/customer.test.csv")

# 사용자 코딩
# 1. 데이터 유형 팡가
print(train.info())
print(test.info())
print(train.shape)
print(test.shape)









