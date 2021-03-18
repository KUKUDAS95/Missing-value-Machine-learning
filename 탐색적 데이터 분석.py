import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

data = pd.read_csv("/home/hong/Downloads/diabetes.csv")
# data = pd.read_csv("원본_nan.csv")
## 안의 아웃라인은 값에 대한 평균 수치
sns.boxplot(x="Outcome", y="BMI", data=data)
# sns.boxplot(y="BMI", data=df)#전체데이터에 대한 임신 횟수
# sns.boxplot(x="Outcome", y="Pregnancies", data=df)#당뇨병 여부에 따른 임신 횟수

# plt.figure(figsize=(12,6))
# sns.barplot(x="Age", y="Outcome", data=df)# 나이대별 발병 여부

# plt.figure(figsize=(12,6))
# sns.barplot(x="Pregnancies", y="Outcome", data=df)#임신 횟수별  발병율
# plt.show()
data_copy=data.copy()
def remove_outlier(d_cp, column):
    column_data=d_cp[d_cp['Outcome']==0][column]
    quan_25 = np.percentile(column_data.values, 25)
    quan_75 = np.percentile(column_data.values, 75)

    iqr = quan_75-quan_25
    iqr=iqr*1.5
    low=quan_25-iqr
    high=quan_75+iqr
    outlier_index = column_data[(column_data<low)|{column_data>high}].index
    print(len(outlier_index))
    d_cp.drop(outlier_index, axis =0, inplace=True)
    print(d_cp.shpae)
    return d_cp

data_copy=remove_outlier(data_copy, 'BMI')

