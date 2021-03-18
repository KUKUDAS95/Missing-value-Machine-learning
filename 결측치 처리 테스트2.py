import pandas as pd
from sklearn import preprocessing

df = pd.read_csv('원본_nan.csv')
df.fillna(df.median(),inplace=True)
print(df['Pregnancies'])
# df = pd.read_csv('/home/hong/Downloads/비정상,정상평균.csv')
# x=df.values.astype(float)
# min_mas_scaler=preprocessing.MinMaxScaler()
# x_scaled=min_mas_scaler.fit_transform(x)
# df=pd.DataFrame(x_scaled,columns=df.columns)
# print(df)
#
df.to_csv('median()값.csv', index=False, encoding='cp949', mode='w')