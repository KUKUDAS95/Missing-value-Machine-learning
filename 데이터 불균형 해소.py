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

####Data info = Pregnancies / Glucose / BloodPressure / SkinThickness / Insulin / BMI / DiabetesPedigreeFunction / Age / Outcome


#link:https://john-analyst.medium.com/smote%EB%A1%9C-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EB%B6%88%EA%B7%A0%ED%98%95-%ED%95%B4%EA%B2%B0%ED%95%98%EA%B8%B0-5ab674ef0b32
df=pd.read_csv('/home/hong/Downloads/diabetes.csv')


x = df.iloc[:,:-1]
y = df.iloc[:,-1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, shuffle=True)

smote = SMOTE(random_state=0)
x_train_over,y_train_over = smote.fit_sample(x_train,y_train)
print('SMOTE 적용 전 학습용 피처/레이블 데이터 세트: ', x_train.shape, y_train.shape)
print('SMOTE 적용 전 레이블 값 분포: \n', pd.Series(y_train).value_counts())
print('SMOTE 적용 후 학습용 피처/레이블 데이터 세트: ', x_train_over.shape, y_train_over.shape)
print('SMOTE 적용 후 레이블 값 분포: \n', pd.Series(y_train_over).value_counts())

over_data=pd.concat([x_train_over,y_train_over],axis=1)#데이터합침
# print(over_data)


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

#===SMOTE적용 후 모델 학습===
print("LogisticRegression")
lr = LogisticRegression(solver='lbfgs',max_iter=400)
modeling(lr,x_train,x_test,y_train,y_test)
print("LGBMClassifier")
lgb = LGBMClassifier(n_estimators=1000,num_leaves=64,n_jobs=-1,boost_from_average=False)#is_unbalance = True
modeling(lgb,x_train,x_test,y_train,y_test)
print("----------------------------------------------")
print("LogisticRegression")
lr = LogisticRegression(solver='lbfgs',max_iter=400)#solver='lbfgs',max_iter=400
modeling(lr,x_train_over,x_test,y_train_over,y_test)
print("LGBMClassifier")
lgb = LGBMClassifier(n_estimators=1000,num_leaves=64,is_unbalance = True,n_jobs=-1,boost_from_average=False)
modeling(lgb,x_train_over,x_test,y_train_over,y_test)


#===이상치제거====
quartile_1 = over_data.quantile(0.25)
quartile_3 = over_data.quantile(0.75)
IQR = quartile_3 - quartile_1
condition = (over_data < (quartile_1 - 1.5 * IQR)) | (over_data > (quartile_3 + 1.5 * IQR))
condition = condition.any(axis=1)
search_df = over_data[condition]
print("----------------------------------------------")
print("이상치제거개수:", len(search_df))
over_data=over_data.drop(search_df.index, axis=0)
# 데이터 저장
# over_data.to_csv('/home/hong/Downloads/이상치제거데이터.csv', index=False, encoding='cp949', mode='w')
print("이상치제거후 데이터개수:",len(over_data))
print("----------------------------------------------")
print("결측치 처리중....")

#0값 -> nan값으로 변경
glu = over_data.replace({'Glucose': 0.0}, {'Glucose': np.NaN})
blu = glu.replace({'BloodPressure': 0.0}, {'BloodPressure':np.NaN})
ski = blu.replace({'SkinThickness': 0.0}, {'SkinThickness':np.NaN})
ins = ski.replace({'Insulin':0},{'Insulin':np.NaN})
bmi = ins.replace({'BMI':0.0},{'BMI':np.NaN})
print("Glocose nan개수:",bmi['Glucose'].isna().sum())
print("BloodPressure nan개수:",bmi['BloodPressure'].isna().sum())
print("SkinThickness nan개수:",bmi['SkinThickness'].isna().sum())
print("Insulin nan개수:",bmi['Insulin'].isna().sum())
print("BMI nan개수:",bmi['BMI'].isna().sum())

#결측치 처리1
data=bmi.dropna(axis=0, how='any')
# data = bmi.interpolate(method='linear', limit_direction='both')
# data = data.interpolate(method='polynomial', order=7)

#결측치 처리2
# trans=SimpleImputer()
# trans.fit(bmi)
# data=trans.transform(bmi)
# print(data[:,4])
# print(np.shape(data))


print("Glocose nan개수:",data['Glucose'].isna().sum())
print("BloodPressure nan개수:",data['BloodPressure'].isna().sum())
print("SkinThickness nan개수:",data['SkinThickness'].isna().sum())
print("Insulin nan개수:",data['Insulin'].isna().sum())
print("BMI nan개수:",data['BMI'].isna().sum())
print("----------------------------------------------")

#데이터 분리
x = data.iloc[:,:-1]
y = data.iloc[:,-1]

# x = data[:,:-1]
# y = data[:,-1]


#정규화 작업
# normalization_df = (x - x.mean())/x.std()
normalization_df2 = (x - x.mean())/(x.max()-x.min())


x_train_outliear, x_test_outliear, y_train_outliear, y_test_outliear = train_test_split(normalization_df2, y, test_size=0.2, stratify=y, shuffle=True)

print("LogisticRegression")
lr = LogisticRegression(solver='lbfgs',max_iter=400)#solver='lbfgs',max_iter=400
modeling(lr,x_train_outliear,x_test_outliear,y_train_outliear,y_test_outliear)

print("LGBMClassifier")
def lgbm_cv(learning_rate, num_leaves, max_depth, min_child_weight, colsample_bytree, feature_fraction, bagging_fraction, lambda_l1, lambda_l2):
    model = lgbm.LGBMClassifier(learning_rate=learning_rate,
                                n_estimators=300,
                                #boosting = 'dart',
                                num_leaves = int(round(num_leaves)),
                                max_depth = int(round(max_depth)),
                                min_child_weight = int(round(min_child_weight)),
                                colsample_bytree = colsample_bytree,
                                feature_fraction = max(min(feature_fraction, 1), 0),
                                bagging_fraction = max(min(bagging_fraction, 1), 0),
                                lambda_l1 = max(lambda_l1, 0),
                                lambda_l2 = max(lambda_l2, 0)
                               )
    scoring = {'roc_auc_score': make_scorer(roc_auc_score)}
    result = cross_validate(model, x_train_outliear, y_train_outliear, cv=5, scoring=scoring)
    auc_score = result["test_roc_auc_score"].mean()
    return auc_score

pbounds = {'learning_rate' : (0.0001, 0.05),
           'num_leaves': (300, 600),
           'max_depth': (2, 25),
           'min_child_weight': (30, 100),
           'colsample_bytree': (0, 0.99),
           'feature_fraction': (0.0001, 0.99),
           'bagging_fraction': (0.0001, 0.99),
           'lambda_l1' : (0, 0.99),
           'lambda_l2' : (0, 0.99),
          }
lgbmBO = BayesianOptimization(f = lgbm_cv, pbounds = pbounds, verbose = 2, random_state = 0 )
lgbmBO.maximize(init_points=5, n_iter = 20, acq='ei', xi=0.01)
fit_lgbm = lgbm.LGBMClassifier(learning_rate=lgbmBO.max['params']['learning_rate'],
                               num_leaves = int(round(lgbmBO.max['params']['num_leaves'])),
                               max_depth = int(round(lgbmBO.max['params']['max_depth'])),
                               min_child_weight = int(round(lgbmBO.max['params']['min_child_weight'])),
                               colsample_bytree=lgbmBO.max['params']['colsample_bytree'],
                               feature_fraction = max(min(lgbmBO.max['params']['feature_fraction'], 1), 0),
                               bagging_fraction = max(min(lgbmBO.max['params']['bagging_fraction'], 1), 0),
                               lambda_l1 = lgbmBO.max['params']['lambda_l1'],
                               lambda_l2 = lgbmBO.max['params']['lambda_l2']
                               )
model = fit_lgbm.fit(x_train_outliear, y_train_outliear)
pred = model.predict(x_test_outliear)
accuracy = accuracy_score(y_test_outliear, pred)
precision = precision_score(y_test_outliear, pred)
recall = recall_score(y_test_outliear, pred)
f1 = f1_score(y_test_outliear, pred)
roc_score = roc_auc_score(y_test_outliear, pred, average='macro')
print('정확도 : {0:.2f}, 정밀도 : {1:.2f}, 재현율 : {2:.2f}'.format(accuracy, precision, recall))
print('f1-score : {0:.2f}, auc : {1:.2f}'.format(f1, roc_score, recall))


#원본
# lgb = LGBMClassifier(n_estimators=100,num_leaves=64,is_unbalance = True,n_jobs=-1,boost_from_average=False)
# modeling(lgb,x_train_outliear,x_test_outliear,y_train_outliear,y_test_outliear)



print("DecisionTree")
dtree= DecisionTreeClassifier(max_depth=5, random_state=3)
modeling(dtree,x_train_outliear,x_test_outliear,y_train_outliear,y_test_outliear)

print("RandomForest")
# rf = RandomForestClassifier(min_samples_leaf=1, max_depth=10,n_estimators=100, random_state=3)
# rf = RandomForestClassifier(min_samples_leaf=1, max_depth=9,n_estimators=80, random_state=7)#테스트 정확도 0.841
rf = RandomForestClassifier(min_samples_leaf=1, max_depth=7,n_estimators=70, random_state=7)#테스트 정확도 0.860
modeling(rf,x_train_outliear,x_test_outliear,y_train_outliear,y_test_outliear)

print("Bagging")
bg=BaggingClassifier(base_estimator=dtree, n_estimators=100, random_state=5)
modeling(bg,x_train_outliear,x_test_outliear,y_train_outliear,y_test_outliear)

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

print("===교차검증===")
#link: https://3months.tistory.com/321
kfold=KFold(n_splits=50, shuffle=True, random_state=7)
cv_lr=cross_val_score(lr, normalization_df2, y, cv=kfold)
cv_lgb=cross_val_score(lgb, normalization_df2, y, cv=kfold)
cv_dt=cross_val_score(dtree, normalization_df2, y, cv=kfold)
cv_rf=cross_val_score(rf, normalization_df2, y, cv=kfold)
cv_bg=cross_val_score(bg, normalization_df2, y, cv=kfold)
print("LogisticRegression 평균:",np.mean(cv_lr))
print("LGBMClassifier 평균:",np.mean(cv_lgb))
print("DecisionTree 평균:",np.mean(cv_dt))
print("RandomForest 평균:",np.mean(cv_rf))
print("Bagging 평균:",np.mean(cv_bg))



#===============이상치제거 그래프===============================

# plt.subplot(521) #세로/가로/위치
# sns.boxplot(data=over_data, x="Outcome", y="Glucose")
#
# plt.subplot(522)
# sns.boxplot(data=search_df, x="Outcome", y="Glucose")
#
# plt.subplot(523)
# sns.boxplot(data=over_data, x="Outcome", y="BloodPressure")
#
# plt.subplot(524)
# sns.boxplot(data=search_df, x="Outcome", y="BloodPressure")
#
# plt.subplot(525)
# sns.boxplot(data=over_data, x="Outcome", y="SkinThickness")
#
# plt.subplot(526)
# sns.boxplot(data=search_df, x="Outcome", y="SkinThickness")
#
# plt.subplot(527)
# sns.boxplot(data=over_data, x="Outcome", y="Insulin")
#
# plt.subplot(528)
# sns.boxplot(data=search_df, x="Outcome", y="Insulin")
#
# plt.subplot(521)
# sns.boxplot(data=over_data, x="Outcome", y="BMI")
#
# plt.subplot(522)
# sns.boxplot(data=search_df, x="Outcome", y="BMI")
# plt.show()