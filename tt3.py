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

# glu = df.replace({'Glucose': 0.0}, {'Glucose': np.NaN})
# blu = glu.replace({'BloodPressure': 0.0}, {'BloodPressure':np.NaN})
# ski = blu.replace({'SkinThickness': 0.0}, {'SkinThickness':np.NaN})
# ins = ski.replace({'Insulin':0},{'Insulin':np.NaN})
# bmi = ins.replace({'BMI':0.0},{'BMI':np.NaN})

x = df.iloc[:, :-1]
y = df.iloc[:, -1]

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
   n_iterations = 100
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




