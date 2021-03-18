from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# df = pd.read_csv('/home/hong/Downloads/diabetes.csv')
df = pd.read_csv('원본_nan.csv')
# print(df)
print(df.isnull().sum())#결측치 확인
# print(df.describe())#데이터프레임의 기초 통계량

# cols = df.columns
# # print(df.shape)
# for col in cols:
#     mean = df[col].mean()
#     std = df[col].std()
#     threshold = mean + 3 * std
#     n_outlier = np.sum(df[col] > threshold)
#     print(col + ". num of outlier : "+str(n_outlier)) #값 > 평균 + 3 * 표준편차들은 아웃라이어

# cols = df.columns
# print("before drop outlier : {}".format(df.shape))
# for col in cols:
#     mean = df[col].mean()
#     std = df[col].std()
#     threshold = mean + 3 * std
#     n_outlier = np.sum(df[col] > threshold)
#     #print(df[df[col] > threshold])
#     df.drop(df[df[col] > threshold].index[:], inplace=True)
#
# df.dropna()
# print("after drop outlier : {}".format(df.shape))#조건을 넘는 값을 아웃라이어로 판단하고, 제거 -> 제거한 결과 768개 에서 727개로 줄어듦
#
# X = df.loc[:, df.columns != "Outcome"]
# y = df.loc[:, df.columns == "Outcome"]
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# print(X_scaled[:,:6])#모든 피쳐 데이터들을 표준화 시켜주기 위해 Standard Scaler를 사용  -> 모든 데이터들이 표준 정규분포를 따르는 값들로 변환









