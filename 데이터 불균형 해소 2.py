from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from bayes_opt import BayesianOptimization
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from bayes_opt import BayesianOptimization
import lightgbm as lgbm
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import cross_validate
from sklearn.model_selection import learning_curve
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score
import datawig
from missingpy import MissForest
from eli5.sklearn import permutation_importance
import eli5
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



####Data info = Pregnancies / Glucose / BloodPressure / SkinThickness / Insulin / BMI / DiabetesPedigreeFunction / Age / Outcome


#link:https://john-analyst.medium.com/smote%EB%A1%9C-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EB%B6%88%EA%B7%A0%ED%98%95-%ED%95%B4%EA%B2%B0%ED%95%98%EA%B8%B0-5ab674ef0b32

df=pd.read_csv('/home/hong/Downloads/diabetes.csv')

#0값 -> nan값으로 변경
print("0값 -> nan값으로 변경")
glu = df.replace({'Glucose': 0.0}, {'Glucose': np.NaN})
blu = glu.replace({'BloodPressure': 0.0}, {'BloodPressure':np.NaN})
ski = blu.replace({'SkinThickness': 0.0}, {'SkinThickness':np.NaN})
ins = ski.replace({'Insulin':0},{'Insulin':np.NaN})
bmi = ins.replace({'BMI':0.0},{'BMI':np.NaN})
print("Glocose nan개수:",bmi['Glucose'].isna().sum())
print("BloodPressure nan개수:",bmi['BloodPressure'].isna().sum())
print("SkinThickness nan개수:",bmi['SkinThickness'].isna().sum())
print("Insulin nan개수:",bmi['Insulin'].isna().sum())
print("BMI nan개수:",bmi['BMI'].isna().sum())

print("결측치 처리")
# data=bmi.dropna(axis=0, how='any')

# data = bmi.interpolate(method='linear', limit_direction='both')
# data = data.interpolate(method='polynomial', order=7)

#결측치 처리2
data = datawig.SimpleImputer.complete(bmi)



#결측치 처리3
# imputer=MissForest()
# imputer.fit(bmi)
# np_imputer=imputer.transform(bmi)
# data=pd.DataFrame(np_imputer,columns=['Pregnancies' , 'Glucose' , 'BloodPressure' , 'SkinThickness' , 'Insulin' , 'BMI' , 'DiabetesPedigreeFunction' , 'Age' , 'Outcome'])
#



print("Glocose nan개수:",data['Glucose'].isna().sum())
print("BloodPressure nan개수:",data['BloodPressure'].isna().sum())
print("SkinThickness nan개수:",data['SkinThickness'].isna().sum())
print("Insulin nan개수:",data['Insulin'].isna().sum())
print("BMI nan개수:",data['BMI'].isna().sum())
print("----------------------------------------------")

x = data.iloc[:,:-1]
y = data.iloc[:,-1]



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, shuffle=True)

smote = SMOTE(random_state=0)
x_train_over,y_train_over = smote.fit_sample(x_train,y_train)
print('SMOTE 적용 전 학습용 피처/레이블 데이터 세트: ', x_train.shape, y_train.shape)
print('SMOTE 적용 전 레이블 값 분포: \n', pd.Series(y_train).value_counts())
print('SMOTE 적용 후 학습용 피처/레이블 데이터 세트: ', x_train_over.shape, y_train_over.shape)
print('SMOTE 적용 후 레이블 값 분포: \n', pd.Series(y_train_over).value_counts())

smote_data=pd.concat([x_train_over,y_train_over],axis=1)#데이터합침



#모델링
def modeling(model,x_train,x_test,y_train,y_test):
    model.fit(x_train,y_train)
    pred = model.predict(x_test)
    metrics(y_test,pred)

#평가 지표
def metrics(y_test,pred):
    accuracy = accuracy_score(y_test,pred)
    precision = precision_score(y_test,pred)
    recall = recall_score(y_test,pred)
    f1 = f1_score(y_test,pred)
    roc_score = roc_auc_score(y_test,pred,average='macro')
    print('정확도 : {0:.2f}, 정밀도 : {1:.2f}, 재현율 : {2:.2f}'.format(accuracy,precision,recall))
    print('f1-score : {0:.2f}, auc : {1:.2f}'.format(f1,roc_score,recall))



#===이상치제거====
quartile_1 = smote_data.quantile(0.25)
quartile_3 = smote_data.quantile(0.75)
IQR = quartile_3 - quartile_1
condition = (smote_data < (quartile_1 - 1.5 * IQR)) | (smote_data > (quartile_3 + 1.5 * IQR))
condition = condition.any(axis=1)
search_df = smote_data[condition]
print("----------------------------------------------")
print("이상치제거개수:", len(search_df))
smote_data2=smote_data.drop(search_df.index, axis=0)
# 데이터 저장
# over_data.to_csv('/home/hong/Downloads/이상치제거데이터.csv', index=False, encoding='cp949', mode='w')
print("이상치제거후 데이터개수:",len(smote_data2))
print("----------------------------------------------")

#데이터 분리
x = smote_data.iloc[:,:-1]
y = smote_data.iloc[:,-1]

#정규화 작업
# normalization_df = (x - x.mean())/x.std()
# normalization_df2=StandardScaler (). fit_transform (x)
normalization_df2 = (x - x.mean())/(x.max()-x.min())


x_train_outliear, x_test_outliear, y_train_outliear, y_test_outliear = train_test_split(normalization_df2, y, test_size=0.2, stratify=y, shuffle=True)


print("RandomForest")

#===Hyperparameter Tuning===
# print("Random Search")
# param_distributions=[{'n_estimators':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],'random_state':[None,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],'max_depth':[1,2,3,4,5,6,7,8,9,10],'min_samples_leaf':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]}]
# randomized_search = RandomizedSearchCV(rf, param_distributions=param_distributions, n_iter=100, return_train_score=True)
# randomized_search.fit(x_train_outliear, y_train_outliear)
# clf2 = randomized_search.best_estimator_
# clf2.fit(x_train_outliear,y_train_outliear)
# print(randomized_search.best_score_)
# print(randomized_search.best_params_)
# print('테스트 정확도: %.3f' % clf2.score(x_test_outliear, y_test_outliear))
#
# print("Grid Search")
# pram_grid=[{'n_estimators':[1,2,3,4,5,6,7,8,9,10,11],'random_state':[None,1,2,3,4,5,6,7,8,9,10],'max_depth':[1,2,3,4,5,6,7,8,9,10],'min_samples_leaf':[1,2,3,4,5,6,7,8,9,10]}]
# gs=GridSearchCV(estimator=rf,param_grid=pram_grid,scoring='accuracy',cv=2)
# gs2=gs.fit(x_train_outliear,y_train_outliear)
# clf = gs.best_estimator_
# clf.fit(x_train_outliear,y_train_outliear)
# print(gs.best_score_)
# print(gs.best_params_)
# print('테스트 정확도: %.3f' % clf.score(x_test_outliear, y_test_outliear))

def rfc_optimization(cv_splits):
    def function(n_estimators, max_depth, min_samples_split):
        return cross_val_score(
            RandomForestClassifier(
                n_estimators=int(max(n_estimators, 0)),
                max_depth=int(max(max_depth, 1)),
                min_samples_split=int(max(min_samples_split, 2)),
                n_jobs=-1,
                random_state=42,
                class_weight="balanced"),
            X=x_train_outliear,
            y=y_train_outliear,
            cv=cv_splits,
            scoring="roc_auc",
            n_jobs=-1).mean()

    parameters = {"n_estimators": (10, 1000),
                  "max_depth": (1, 150),
                  "min_samples_split": (2, 10)}

    return function, parameters


def bayesian_optimization(function, parameters):
   n_iterations = 24
   gp_params = {"alpha": 0.01} #  1e-4}

   BO = BayesianOptimization(function, parameters)
   BO.maximize(n_iter=n_iterations, **gp_params)

   bk = BO.res

   n_estimators = []
   max_depth = []
   min_samples_split = []
   target = []

   for items in bk:
       item = items['params']

       n_estimators.append(item['n_estimators'])
       max_depth.append(item['max_depth'])
       min_samples_split.append(item['min_samples_split'])
       target.append(items['target'])

   n_estimators = np.array(n_estimators)
   max_depth = np.array(max_depth)
   min_samples_split = np.array(min_samples_split)
   target = np.array(target)

   print(n_estimators.shape, max_depth.shape, min_samples_split.shape, target.shape)

   ne = np.median(n_estimators)
   md = np.median(max_depth)
   ms = np.median(min_samples_split)
   print('n_estimators:', ne, 'max_depth: ', md, 'min_samples_split: ', ms)
   print("BO.max:",BO.max)

   myBO = {'params': {'n_estimators': ne, 'max_depth': md, 'min_samples_split': ms}}

   return BO.max


def train(X_train, y_train, function, parameters):
    best_solution = bayesian_optimization(function, parameters)

    params = best_solution["params"]
    print(params)
    model = RandomForestClassifier(
        n_estimators=int(max(params["n_estimators"], 0)),
        max_depth=int(max(params["max_depth"], 1)),
        min_samples_split=int(max(params["min_samples_split"], 2)),
        n_jobs=-1,
        random_state=42,

        class_weight="balanced")

    model.fit(X_train, y_train)

    return model


function, parameters = rfc_optimization(4)
classifier = train(x_train_outliear, y_train_outliear, function, parameters)

y_pred = classifier.predict(x_test_outliear)
cm = confusion_matrix(y_test_outliear, y_pred)

accuracy_score = accuracy_score(y_test_outliear, y_pred)
precision = precision_score(y_test_outliear, y_pred)
recall = recall_score(y_test_outliear, y_pred)
f1 = f1_score(y_test_outliear, y_pred)
roc_score = roc_auc_score(y_test_outliear, y_pred, average='macro')

print('정확도 : {0:.2f}, 정밀도 : {1:.2f}, 재현율 : {2:.2f}'.format(accuracy_score, precision, recall))
print('f1-score : {0:.2f}, auc : {1:.2f}'.format(f1, roc_score, recall))

#중요변수 테스트!! 에러남 ㅎㅎ
rf_im=classifier.feature_importances_
# index=['Pregnancies' , 'Glucose' , 'BloodPressure' , 'SkinThickness' , 'Insulin' , 'BMI' , 'DiabetesPedigreeFunction' , 'Age']
#
# rf_im=pd.Series(rf_im,index)
# plt.show(rf_im)





