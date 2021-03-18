from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM
import pandas as pd
from urllib import request
import numpy as np
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

####Data info = Pregnancies / Glucose / BloodPressure / SkinThickness / Insulin / BMI / DiabetesPedigreeFunction / Age / Outcome

# 데이터 세트의 URL을 설정
# dataset = pd.read_csv("/home/hong/Downloads/average_data.csv")
# dataset= pd.read_csv("delete_missing value.csv")
# dataset= pd.read_csv("interpolate_missing value.csv")
# dataset = pd.read_csv("/home/hong/Downloads/tttt.csv")
# url = "http://nrvis.com/data/mldata/pima-indians-diabetes.csv"
# f = request.urlopen(url)

# random seed for reproducibility
np.random.seed(2)

# 데이터 세트를 불러옵니다.
# dataset = np.loadtxt(f, delimiter=",")

col = list(map(str, dataset.columns))
X=dataset[col[:-1]]
Y=dataset[col[-1]]

X = np.array(X)
Y = np.array(Y)

# X = dataset[:,0:8]
# Y = dataset[:,8]
# print(X[0]) #768/8
# print(Y.shape)

print('-------x reshape-----------')
X = X.reshape((X.shape[0], X.shape[1], 1)) # (4,3,1) reshape 전체 곱 수 같아야 4*3=4*3*1
print('x shape : ', X.shape)


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, stratify=Y, shuffle=True)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, stratify=y_test, shuffle=True)

# # 2. 모델 구성
model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(8, 1)))
# DENSE와 사용법 동일하나 input_shape=(열, 몇개씩잘라작업)
model.add(Dense(5))
model.add(Dense(1))

model.summary()

# # 3. 실행
model.compile(optimizer='adam', loss='mse',metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto')
model.fit(x_train, y_train, epochs=100, batch_size=10, verbose=2, validation_data=(x_val, y_val), callbacks=[early_stopping])

Xtrain_pred = model.evaluate(x_train, y_train, batch_size=10)
print('Train loss and acc : ', Xtrain_pred)


y_pred = model.predict(x_test)

print(y_test.shape)

# label = np.sum(y_test), axis=-1)
print(classification_report(y_test, np.argmax(y_pred, axis=1)))
print(confusion_matrix(y_test, np.argmax(y_pred, axis=1)))



