import pandas as pd
import numpy as np
pd.set_option('display.max_columns',50)
import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/nba/nba.csv",encoding='latin',sep=';')
df.head()


# 104. Tm 컬럼은 각 팀의 이름을 의미한다. TOR팀의 평균나이를 소수 4째 자리까지 구하여라

# df.info()
# df.Tm.info()
# df.Tm

tor_age = df[df.Tm == 'TOR'].Age.mean()
print(round(tor_age, 4))
print(np.round(tor_age, 4))

df[df.index.isin(df.Player.drop_duplicates().index)].Age.sort_values() # 19

df[df.Age == 19].Pos.value_counts() # SG

df[['FirstName', 'LastName']] = df['Player'].str.split(' ', n=1, expand=True)
df.FirstName

df['Player'].str.split().str[0].str.lower().value_counts().head(1)

df.groupby('Pos')['PTS'].mean()

df[df.index.isin(df.groupby('Tm')['G'].idxmax())][['Player']].value_counts()
df.loc[df.groupby('Tm')['G'].idxmax()][['Player']].value_counts()


df.sort_values(['Tm','G'])
df.sort_values(['Tm','G']).groupby('Tm').tail(1)


# 111. Tm의 값이 MIA이며 Pos는 C또는 PF인 선수의 MP값의 평균은?

# 조건에 맞는 선수 필터링
filtered = df[(df['Tm'] == 'MIA') & (df['Pos'].isin(['C', 'PF']))]

# MP 평균 계산
avg_mp = filtered['MP'].mean()

print(avg_mp)


# 112. Age의 평균 이상인 그룹과 평균 미만인 그룹간의 G값의 평균의 차이는?
df_avg = df.Age.mean()
df_avg

df1 = df[df.Age >= df_avg]
df1.head(1)

df2 = df[df.Age <= df_avg]
df2.head(1)

df_diff = df1.G.mean() - df2.G.mean()
df_diff

# 113. 평균나이가 가장 젊은 팀은 어디인가
df.groupby('Tm')['Age'].mean().sort_values().index[0]
df.groupby('Pos')['MP'].mean()

