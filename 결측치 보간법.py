import pandas as pd
import numpy as np
import csv

# df = pd.read_csv("/home/hong/Downloads/setting.csv")
# df = pd.read_csv("/home/hong/Downloads/tttt.csv")
df = pd.read_csv("/home/hong/Downloads/diabetes.csv")
# df=df.drop(columns="Outcome")

# pre = df.replace({'Pregnancies': 0}, {'Pregnancies': np.NaN})
glu = df.replace({'Glucose': 0.0}, {'Glucose': np.NaN})
blu = glu.replace({'BloodPressure': 0.0}, {'BloodPressure':np.NaN})
ski = blu.replace({'SkinThickness': 0.0}, {'SkinThickness':np.NaN})
ins = ski.replace({'Insulin':0},{'Insulin':np.NaN})
bmi = ins.replace({'BMI':0.0},{'BMI':np.NaN})
# print(bmi)

# col = list(map(str, df.columns))
# print(col)
# x=df[col[:-1]]
# print(x)
# y=df[col[-1]]
# print(y)


# col = list(map(str, df.columns))
# x=df[col[:-1]]
# print(x)

# bmi = df.replace({'Insulin': 0.0}, {'Insulin': np.NaN})


# print(bmi['Insulin'])

#결측치 있는 행 모두 삭제
# data=bmi.dropna(axis=0, how='any')

#결측치에 보간값 채워넣기
# data = bmi.interpolate(method='linear', limit_direction='both')
# data=bmi.interpolate(method='polynomial', order=7)
# print(data)

# print(data['Insulin'])

#정규화
# normalization_df = (bmi - bmi.mean())/bmi.std()
# normalization_df = (x - x.mean())/x.std()
# print(normalization_df.head())
#
# #데이터 저장
bmi.to_csv('원본_nan.csv', index=False, encoding='cp949', mode='w')














