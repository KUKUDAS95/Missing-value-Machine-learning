from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import random

####Data info = Pregnancies / Glucose / BloodPressure / SkinThickness / Insulin / BMI / DiabetesPedigreeFunction / Age / Outcome
# 데이터 세트를 불러옵니다.
data = pd.read_csv("이상치test.csv")
# data = pd.read_csv("/home/hong/Downloads/average_data.csv")
# data = pd.read_csv("/home/hong/Downloads/tttt.csv")
# data= pd.read_csv("interpolate_missing value.csv")
# data= pd.read_csv("delete_missing value.csv")
# data = pd.read_csv("/home/hong/Downloads/diabetes.csv")
# data=data.drop(columns="Outcome")
col = list(map(str, data.columns))
x=data[col[:-1]]
y=data[col[-1]]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, shuffle=True)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, stratify=y_test, shuffle=True)

rf = RandomForestClassifier(n_estimators=100, random_state=random.seed()).fit(x_train, y_train)

print("train score : {}".format(rf.score(x_train, y_train)))
print("val score : {}".format(rf.score(x_val, y_val)))

predict_y = rf.predict(x_test)
print(classification_report(y_test, rf.predict(x_test)))
print("test score : {}".format(rf.score(x_test, y_test)))
