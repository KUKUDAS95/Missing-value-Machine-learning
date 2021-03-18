import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')
# diabetes_data = pd.read_csv("/home/hong/Downloads/원본_혈압.csv")
# diabetes_data = pd.read_csv('/home/hong/Downloads/diabetes.csv')
# diabetes_data = pd.read_csv('/home/hong/Downloads/비정상,정상평균.csv')
diabetes_data = pd.read_csv("test_imputed.csv")
# diabetes_data = pd.read_csv("/home/hong/Downloads/tttt.csv")
# diabetes_data= pd.read_csv("delete_missing value.csv")
# diabetes_data = pd.read_csv("/home/hong/Downloads/average_data.csv")
print(diabetes_data['Outcome'].value_counts()) # 타겟값이 어떻게 되어있는지 확인해야함
diabetes_data.head(3)

# count, bin_dividers = np.histogram(diabetes_data['BMI'], bins=5)
# # display(bin_dividers)
# bin_names = ['저체중', '보통체중', '경도비만', '비만', '고도비만']
# diabetes_data['BMI_bin'] = pd.cut(x=diabetes_data['BMI'],
#  bins=bin_dividers,
# labels=bin_names,
# include_lowest=True)
# BMI_dummies = pd.get_dummies(diabetes_data['BMI_bin'])
# diabetes_data['BMI_bin'] = BMI_dummies
# diabetes_data.drop('BMI', axis=1, inplace=True)
# diabetes_data.head()
# # print(BMI_dummies)
#
# count, bin_dividers = np.histogram(diabetes_data['BloodPressure'], bins=4)
# # display(bin_dividers)
# bin_names = ['저혈압', '정상', '고혈압', '초고혈압']
# diabetes_data['BloodPressure_bin'] = pd.cut(x=diabetes_data['BloodPressure'],
#  bins=bin_dividers,
# labels=bin_names,
# include_lowest=True)
# BloodPressure_dummies = pd.get_dummies(diabetes_data['BloodPressure_bin'])
# diabetes_data['BloodPressure_bin'] = BloodPressure_dummies
# diabetes_data.drop('BloodPressure', axis=1, inplace=True)
# diabetes_data.head()

def get_clf_eval(y_test, pred): # 매개변수값에 None 이라고 적힌것은 특별히 의미있는 값은 아님
 confusion = confusion_matrix(y_test, pred)
 accuracy = accuracy_score(y_test, pred)
 precision = precision_score(y_test, pred)
 recall = recall_score(y_test, pred)
 f1 = f1_score(y_test, pred)

 print('오차행렬')
 print(confusion)
 print()
 print('정확도 : {0:.4f}, 정밀도 {1:.4f}, 재현율 : {2:.4f}, F1:{3:.4f}'.format(accuracy, precision, recall, f1))
#독립변수, 종속변수 나누기
y_df = diabetes_data.Outcome
X_df = diabetes_data.drop('Outcome', axis=1)
#정규화 작업
scaler = StandardScaler()
data_scaled = scaler.fit_transform(X_df)
# 학습용 평가용 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(data_scaled, y_df, test_size=0.2, random_state=1)
# 분류기 객체 생성
lr_clf = LogisticRegression()
lr_clf.fit(X_train , y_train)
pred = lr_clf.predict(X_test)
get_clf_eval(y_test , pred)