from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import pandas as pd

# 데이터를 로딩하고 학습데이타와 테스트 데이터 분리
data = pd.read_csv("/home/hong/Downloads/diabetes.csv")
col = list(map(str, data.columns))
x_data=data[col[:-1]]
y_data=data[col[-1]]

# print(x_data.shape) #(6, 2)
# print(y_data.shape) #(6,)

####################

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2,stratify=y_data, shuffle=True, random_state=121)
dtree = DecisionTreeClassifier()

### parameter 들을 dictionary 형태로 설정
# parameters = {'max_depth':[1,2,3], 'min_samples_split':[2,3]}
parameters = {'criterion':['gini','entropy'], 'max_depth':[None,2,3,4,5,6], 'max_leaf_nodes':[None,2,3,4,5,6,7], 'min_samples_split':[2,3,4,5,6], 'min_samples_leaf':[1,2,3], 'max_features':[None,'sqrt','log2',3,4,5]}


# param_grid의 하이퍼 파라미터들을 3개의 train, test set fold 로 나누어서 테스트 수행 설정.
### refit=True 가 default 임. True이면 가장 좋은 파라미터 설정으로 재 학습 시킴.
# grid_dtree = GridSearchCV(dtree, param_grid=parameters, cv=3, refit=True)
grid_dtree = GridSearchCV(dtree, param_grid=parameters, cv=3, refit=True,)



# 붓꽃 Train 데이터로 param_grid의 하이퍼 파라미터들을 순차적으로 학습/평가 .
grid_dtree.fit(x_train, y_train)

# GridSearchCV 결과 추출하여 DataFrame으로 변환
pd.set_option('display.max_row', None)
pd.set_option('display.max_columns', None)
scores_df = pd.DataFrame(grid_dtree.cv_results_)
scores_df[['params', 'mean_test_score', 'rank_test_score', \
           'split0_test_score', 'split1_test_score', 'split2_test_score']]
scores_df.to_csv("/home/hong/Downloads/asdf.csv")