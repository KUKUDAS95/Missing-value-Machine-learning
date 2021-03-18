import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df=pd.read_csv('/home/hong/Downloads/diabetes.csv')
# sns.boxplot(x="Outcome", y="BMI", data=df)
plt.show()
# df = pd.read_csv("test_imputed.csv")

def get_outlier(df, column, weight=1.5):
    # target 값과 상관관계가 높은 열을 우선적으로 진행
    column_data=df[df['Outcome']==1][column]
    quantile_25 = np.percentile(column_data.values, 25)
    quantile_75 = np.percentile(column_data.values, 75)

    IQR = quantile_75 - quantile_25
    IQR_weight = IQR * weight

    lowest = quantile_25 - IQR_weight
    highest = quantile_75 + IQR_weight

    outlier_idx = df[column][(df[column] < lowest) | (df[column] > highest)].index
    return outlier_idx


# 함수 사용해서 이상치 값 삭제
outlier_idx = get_outlier(df=df, column='BMI', weight=1.5)
df.drop(outlier_idx, axis=0, inplace=True)
sns.boxplot(x="Outcome", y="BMI", data=df)
print(df)
plt.show()
# df.to_csv('이상치test2.csv', index=False, encoding='cp949', mode='w')
