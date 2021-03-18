import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


# data = pd.read_csv("/home/hong/Downloads/diabetes.csv")
data = pd.read_csv("mean()값.csv")


col = list(map(str, data.columns))
x_data=data[col[:-1]]
y_data=data[col[-1]]

print(x_data.shape) #(6, 2)
print(y_data.shape) #(6,)

####################

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2,stratify=y_data, shuffle=True)

estimator = DecisionTreeClassifier()

param_grid = {'criterion':['gini'], 'max_depth':[None,2,3,4,5,6]}
param_grid = {'criterion':['gini','entropy'], 'max_depth':[None,2,3,4,5,6], 'max_leaf_nodes':[None,2,3,4,5,6,7], 'min_samples_split':[2,3,4,5,6], 'min_samples_leaf':[1,2,3], 'max_features':[None,'sqrt','log2',3,4,5]}

grid = GridSearchCV(estimator, param_grid=param_grid)
grid = GridSearchCV(estimator, param_grid=param_grid, cv=3, scoring='accuracy') #디폴트로 cv=3, 분류에서 디폴트로 scoring='accuracy'

grid.fit(x_data, y_data)

print(grid.best_score_)
print(grid.best_params_)
df = pd.DataFrame(grid.cv_results_)
print(df)
#print(df.sort_values(by='param_max_depth'))
#print(df.sort_values(by='param_max_depth', ascending=0))
#print(df.sort_values(by='rank_test_score'))

# #'''
# estimator = grid.best_estimator_
# #'''
# '''
# #estimator = KNeighborsClassifier(**grid.best_params_)
# estimator = KNeighborsClassifier()
# estimator.set_params(**grid.best_params_)
#
# estimator.fit(x_data, y_data)
# '''
#
# print(x_data[:2])
# '''
# [[3 4]
#  [7 5]]
# '''
# print(y_data[:2]) #[1 1]
# #
# y_predict = estimator.decision_function(x_data)[:2]
# print(y_predict)
# '''
# [-0.3996664  -0.08782625]
# '''
# ##y_predict = estimator.predict_proba(x_data[:2]) #predict_proba is not available when  probability=False
# ##print(y_predict)
# ##'''
# ##[[0.42256931 0.57743069]
# ## [0.43491458 0.56508542]]
# ##'''
# y_predict = estimator.predict(x_data[:2])
# print(y_predict) #[1 1]
#
# #########
#
# '''
# score = grid.score(x_test, y_test)
# print(score)
# '''