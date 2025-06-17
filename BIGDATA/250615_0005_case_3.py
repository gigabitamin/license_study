# 출력을 원할 경우 print() 함수 활용
# 예시) print(df.head())

# getcwd(), chdir() 등 작업 폴더 설정 불필요
# 파일 경로 상 내부 드라이브 경로(C: 등) 접근 불가

import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv("data/bcc.csv")

# 사용자 코딩

# 해당 화면에서는 제출하지 않으며, 문제 풀이 후 답안제출에서 결괏값 제출

print(df.info())
print(df.isnull().sum())


df['log_resistin'] = np.log(df['Resistin'])

group1 = df[df['Classification'] ==1]['log_resistin']
group2 = df[df['Classification'] ==2]['log_resistin']

# 넘파이 사용시 자유도 1 빼주기, 데이터프레임에서는 자동으로 빼서 사용됨
# var1 = np.var(group1, ddof=1) 
# var2 = np.var(group2, ddof=1)

# (1) 검정통계량

var1 = group1.var()
var2 = group2.var()

dof_1 = len(group1)-1
dof_2 = len(group2)-1

# print(dof_1, dof_2)
# 51 63 -> var2 의 자유도가 더 큼 : 분자로 사용

f_stat = var2/var1
print(round(f_stat, 3))


# (2) 합동 분산 추정량
n1 = len(group1)
n2 = len(group2)

pooled_var = ((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2)

print(round(pooled_var, 3))

# (3) 독립표본 t-검정 p값

mean1 = group1.mean()
mean2 = group2.mean()

t_stat = (mean1-mean2) / np.sqrt(pooled_var*(1/n1 + 1/n2))
p_value = 2*(1-stats.t.cdf(abs(t_stat), df=n1+n2-2)) # cdf 누적밀도함수 사용해서 유의수준 a 값 구하기
print(round(p_value, 3))





# ====== ttest_ind 활용 === 

ttest_result = stats.ttest_ind(group1, group2, equal_var=True)
# print(round(ttest_result, 3))
print(ttest_result)


# 기타 ttest_1samp, ttest_rel













