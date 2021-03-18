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
from sklearn.metrics import confusion_matrix
import datawig
from missingpy import MissForest
from eli5.sklearn import permutation_importance
import eli5
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
from scipy import stats
from scipy.stats import zscore
####Data info = Pregnancies / Glucose / BloodPressure / SkinThickness / Insulin / BMI / DiabetesPedigreeFunction / Age / Outcome
from hyperopt import hp

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

print("Glocose nan개수:",data['Glucose'].isna().sum())
print("BloodPressure nan개수:",data['BloodPressure'].isna().sum())
print("SkinThickness nan개수:",data['SkinThickness'].isna().sum())
print("Insulin nan개수:",data['Insulin'].isna().sum())
print("BMI nan개수:",data['BMI'].isna().sum())
print("----------------------------------------------")

x = data.iloc[:,:-1]
y = data.iloc[:,-1]

smote = SMOTE(random_state=0)
print('SMOTE 적용 전 데이터 세트: ', x.shape, y.shape)
print('SMOTE 적용 전 레이블 값 분포: \n', pd.Series(y).value_counts())
x, y = smote.fit_sample(x, y)

print(x.shape, y.shape)
print('SMOTE 적용 후 학습용 데이터 세트: ', x.shape, y.shape)
print('SMOTE 적용 후 레이블 값 분포: \n', pd.Series(y).value_counts())

smote_data=pd.concat([x,y],axis=1)#데이터합침
print(smote_data.shape)

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

x = smote_data2.iloc[:,:-1]
y = smote_data2.iloc[:,-1]
# print(type(x))
#정규화 작업
# x = (x - x.mean())/x.std()
stand_x = StandardScaler().fit_transform(np.array(x))
minma_x = MinMaxScaler().fit_transform(np.array(x))
# normalization_df = (x - x.mean())/(x.max()-x.min())
def norm(x):
    _max = x.max()
    _min = x.min()
    _denom = _max - _min
    return (x - _min) / _denom

x = norm(x)
# normalization_df2 = StandardScaler().fit_transform(x)
# print(normalization_df.shape)
# print(type(normalization_df2))

pca = PCA(n_components=5)  # 주성분을 몇개로 할지 결정
pca_x = x.copy()
pca.fit(pca_x)
pca.transform(pca_x)
pca_x = norm(pca_x)
# principalDf = pd.DataFrame(data=printcipalComponents, columns = ['principal component1', 'principal component2'])# 주성분으로 이루어진 데이터 프레임 구성


# feature = x[ ['Pregnancies' , 'Glucose' , 'BloodPressure' , 'SkinThickness' , 'Insulin' , 'BMI' , 'DiabetesPedigreeFunction' , 'Age'] ]
# print(feature.shape)
model = KMeans(n_clusters=2, algorithm='auto')
model.fit(x)
# print("aa:", feature.shape)
predict = model.predict(x)
# predict.columns = ['Outcome']
# r = pd.concat([feature, predict], axis=1)

# r = pd.merge(feature, predict, how='outer', left_index=True, right_index=True)
feat_x = np.array(x)

predict = np.array(predict)
predict = predict.reshape(-1, 1)

# printcipalComponents = printcipalComponents.reshape(-1, 5)

y = np.array(y)
y = y.reshape(-1, 1)

r = np.hstack([feat_x, predict, y])

df_heatmap = sns.heatmap(data=x.corr(), cbar = True, annot = True, annot_kws={'size' : 6}, fmt = '.2f', square = True, cmap = 'Blues')


plt.show()

# r = pd.DataFrame(r, columns=['Pregnancies' , 'Glucose' , 'BloodPressure' , 'SkinThickness' , 'Insulin' , 'BMI' , 'DiabetesPedigreeFunction' , 'Age', 'Outcome'])
# print(r.head())
# print(predict.shape)
# print(r.shape)
x = r[:, :-1]
y = r[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, shuffle=True)

print("RandomForest")


def rfc_optimization(cv_splits):
    def function(n_estimators, max_depth, min_samples_split, max_features, class_weight, random_state):

        if max_features == 1:
            max_features = 'auto'
        elif max_features == 2:
            max_features = 'sqrt'
        else:
            max_features = 'log2'

        if class_weight == 1:
            class_weight = 'balanced'
        else:
            class_weight = 'balanced_subsample'


        return cross_val_score(
            RandomForestClassifier(
                n_estimators=int(max(n_estimators, 0)),
                max_depth=int(max(max_depth, 1)),
                min_samples_split=int(max(min_samples_split, 2)),
                n_jobs=-1,
                max_features=max_features,
                class_weight=class_weight,
                random_state=int(max(random_state, 0)),),
            X=x_train,
            y=y_train,
            cv=cv_splits,
            scoring="roc_auc",
            n_jobs=-1).mean()

    parameters = {"n_estimators": (10, 200),
                  "max_depth": (2, 50),
                  "min_samples_split": (1, 10),
                  "max_features": (1, 3),
                  "class_weight": (1, 2),
                  "random_state": (1, 70),
                  }

    return function, parameters


def bayesian_optimization(function, parameters):
   n_iterations = 30
   gp_params = {"alpha": 1e-4}

   BO = BayesianOptimization(function, parameters)
   BO.maximize(n_iter=n_iterations, init_points=7, **gp_params)

   # bk = BO.res
   #
   # n_estimators = []
   # max_depth = []
   # min_samples_split = []
   # target = []
   #
   # for items in bk:
   #     item = items['params']
   #
   #     n_estimators.append(item['n_estimators'])
   #     max_depth.append(item['max_depth'])
   #     min_samples_split.append(item['min_samples_split'])
   #     target.append(items['target'])
   #
   # n_estimators = np.array(n_estimators)
   # max_depth = np.array(max_depth)
   # min_samples_split = np.array(min_samples_split)
   # target = np.array(target)
   #
   # print(n_estimators.shape, max_depth.shape, min_samples_split.shape, target.shape)
   #
   # ne = np.median(n_estimators)
   # md = np.median(max_depth)
   # ms = np.median(min_samples_split)
   # print('n_estimators:', ne, 'max_depth: ', md, 'min_samples_split: ', ms)
   # print("BO.max:",BO.max)
   #
   # myBO = {'params': {'n_estimators': ne, 'max_depth': md, 'min_samples_split': ms}}

   return BO.max


def train(X_train, y_train, function, parameters):
    best_solution = bayesian_optimization(function, parameters)
    params = best_solution["params"]

    if int(params['max_features']) == 1:
        params['max_features'] = 'auto'
    elif int(params['max_features']) == 2:
        params['max_features'] = 'sqrt'
    else:
        params['max_features'] = 'log2'

    if int(params['class_weight']) == 1:
        params['class_weight'] = 'balanced'
    else:
        params['class_weight'] = 'balanced_subsample'

    print("params : ", params)

    model = RandomForestClassifier(
        n_estimators=int(max(params["n_estimators"], 0)),
        max_depth=int(max(params["max_depth"], 1)),
        min_samples_split=int(max(params["min_samples_split"], 2)),
        n_jobs=-1,
        max_features=params['max_features'],
        class_weight=params['class_weight'],
        random_state=int(max(params["random_state"], 0)),)

    model.fit(X_train, y_train)

    return model


function, parameters = rfc_optimization(5)
classifier = train(x_train, y_train, function, parameters)

y_pred = classifier.predict(x_test)
cm = confusion_matrix(y_test, y_pred)

accuracy_score = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_score = roc_auc_score(y_test, y_pred, average='macro')

print('정확도 : {0:.2f}, 정밀도 : {1:.2f}, 재현율 : {2:.2f}'.format(accuracy_score, precision, recall))
print('f1-score : {0:.2f}, auc : {1:.2f}'.format(f1, roc_score, recall))




