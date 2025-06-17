
# 빅실 예제 복습

# 유형 3
# 제공된 데이터(bcc.csv)는 암 환자와 정상인의 리지스틴 수치에 대한 자료이며,  
# 두 집단의 로그 리지스틴 값에 차이가 있는지를 검정하려고 한다.  
# 소문항별로 답을 구한 후, 구한 답을 제시된 [제출 형식]에 맞춰 답안 제출 페이지에 입력하시오. (단, 모델은 절편항을 포함한다.)  



import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/refs/heads/master/krdatacertificate/s1.csv')
df.head()


# 1-1) 두 집단의 로그 리지스틴 값의 분산에 차이가 있는지를 알아보기 위해
# F-검정을 수행할 때 검정통계량의 값을 구하여라.
# (단, 분자의 자유도가 분모의 자유도보다 크도록 하여라. 반올림하여 소수 셋째 자리까지 작성)


import numpy as np
cls_1 = df[df['Classification'] == 1]['Resistin']
cls_2 = df[df['Classification'] == 2]['Resistin']

cls_1_log = np.log(cls_1)
cls_2_log = np.log(cls_2)

cls_1_var = np.var(cls_1_log, ddof=1)
cls_2_var = np.var(cls_2_log, ddof=1)

if cls_1_var > cls_2_var:
    f_stat = cls_1_var / cls_2_var
else:
    f_stat = cls_2_var / cls_1_var
    
round_result = round(f_stat, 3)
print(round_result)


# 1-2) 두 집단의 로그 리지스틴 값에 대한 합동 분산 추정량을 구하여라.
# 반올림하여 소수 셋째 자리까지 작성


n1 = len(cls_1)
n2 = len(cls_2)

var_pooled = ((n1-1)*cls_1_var + (n2-1)*cls_2_var) / (n1+n2-2)
print(round(var_pooled, 3))


# 1-3) 2번 문제에서 구한 합동 분산 추정량을 이용하여 두 집단의 로그 리지스틴 값에 유의미한 차이가 있는지 독립표본 t-검정을 수행하고 p-값을 구하여라. (반올림하여 소수 셋째 자리까지 작성)


from scipy import stats

m1 = cls_1_log.mean()
m2 = cls_2_log.mean()

se = var_pooled*(1/n1 + 1/n2)
t_stat = (m1-m2) / np.sqrt(se)
p_v = 2*(1-stats.t.cdf(abs(t_stat), df=n1+n2-2))
print(round(p_v, 3))


# case 2
print(stats.ttest_ind(cls_1_log, cls_2_log)[1])


