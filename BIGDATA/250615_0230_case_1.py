# 출력을 원할 경우 print() 함수 활용
# 예시) print(df.head())

# getcwd(), chdir() 등 작업 폴더 설정 불필요
# 파일 경로 상 내부 드라이브 경로(C: 등) 접근 불가

import pandas as pd

df = pd.read_csv("data/employee_performance.csv")

# 사용자 코딩

# 해당 화면에서는 제출하지 않으며, 문제 풀이 후 답안제출에서 결괏값 제출

# print(df.info())

# 1번
# print(df.isnull().sum())
# df['고객만족도'].fillna(df['고객만족도'].mean(), inplace=True)
df['고객만족도'] = df['고객만족도'].fillna(df['고객만족도'].mean())
# print(df.isnull().sum())

# 2번
# print(df.isnull().sum())

df.drop(df[df['근속연수'].isnull()].index, axis=0, inplace=True)
# print(df.isnull().sum())

# df.dropna(subset=['근속연수'], inplace=True)

# 3번

quantile_3 = df['고객만족도'].quantile(0.75)
# print(int(quantile_3))

print(df['고객만족도'].shape)
print(952*0.75)
print(df['고객만족도'][713])
print(df['고객만족도'].iloc[713])
print(df['고객만족도'].loc[713])


# 4번
# print(int(df.groupby('부서')['연봉'].mean().sort_values(ascending=False).iloc[1]))




