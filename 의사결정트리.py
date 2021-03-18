# _*_ Encoding:UTF-8 _*_ #
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from urllib import request
from sklearn.metrics import classification_report
import graphviz
from sklearn.tree import export_graphviz

####Data info = Pregnancies / Glucose / BloodPressure / SkinThickness / Insulin / BMI / DiabetesPedigreeFunction / Age / Outcome
####Data info ####(독립변수)
# 임신횟수
# 경구용 포도당 내성 테스트에서 혈장 포도당 농도 2시간
# 이완성 혈압(mmHg)
# 삼두근 피부주름 두께(mm)
# 2시간 혈청 인슐린(mu U/ml)
# 체질량지수(weight in kg/(height in m)^2)
# 당뇨병혈통 기능
# 연령
# 결과 전체 768, 1=268개, 0=나머지

# 데이터 세트를 불러옵니다.
# data= pd.read_csv("interpolate_missing value.csv")
# data= pd.read_csv("delete_missing value.csv")
# data = pd.read_csv("/home/hong/Downloads/average_data.csv")
# data = pd.read_csv("test_imputed.csv")
# data = pd.read_csv("원본_nan.csv")
# data = pd.read_csv("이상치test2.csv")
data = pd.read_csv("/home/hong/Downloads/diabetes.csv")
# data = pd.read_csv('mean()값.csv')
# data = data.drop(columns="Outcome")

col = list(map(str, data.columns))
x=data[col[:-1]]
y=data[col[-1]]
# print(x,y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, shuffle=True)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, stratify=y_test, shuffle=True)

# 모델
dtree= DecisionTreeClassifier(max_depth=5, random_state=0).fit(x_train, y_train)#max_depth 설정 과적합 방지
# dtree= DecisionTreeClassifier(max_depth=2, min_samples_leaf=3, criterion='entropy',max_features='sqrt',max_leaf_nodes=2).fit(x_train, y_train)#max_depth 설정 과적합 방지


print("train score : {}".format(dtree.score(x_train, y_train)))
print("val score : {}".format(dtree.score(x_val, y_val)))

predict_y = dtree.predict(x_test)
print(classification_report(y_test, dtree.predict(x_test)))
print("test score : {}".format(dtree.score(x_test, y_test)))
##########################

##시각화##
# export_graphviz(dtree, out_file='tree.dot', class_names=['positive', 'negative'], feature_names=data.columns[:-1], impurity=False, filled=True)
#
# with open('tree1.dot') as file_reader:
#     dot_graph=file_reader.read()
#
# dot = graphviz.Source(dot_graph)
# dot.render(filename='tree1.png')

#원본데이터 비율 맞게 분할 저장
# data = pd.read_csv("/home/hong/Downloads/diabetes.csv")
# col = list(map(str, data.columns))
# y=data[col[-1]]
# data_train, data_test = train_test_split(data, test_size=0.2, stratify=y, shuffle=True)
# data_train.to_csv('/home/hong/Downloads/train.csv', index=False, encoding='cp949', mode='w')
# data_test.to_csv('/home/hong/Downloads/test.csv', index=False, encoding='cp949', mode='w')