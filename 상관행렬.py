import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# df = pd.read_csv("/home/hong/Downloads/diabetes.csv")
# df = pd.read_csv('most_frequent 값.csv')
df = pd.read_csv('median()값.csv')
# df = pd.read_csv('mean()값.csv')
# print(df.describe()) #head, tail, shape, columns, info, describe, isnull, sum, corr

#카이 제곱 통계 테스트값
X = df.iloc[:,0:8]
y = df.iloc[:,-1]
bestfeatures = SelectKBest(score_func=chi2, k=8)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Features','Score']
print(featureScores.nlargest(8,'Score'))

#막대 그래프를 사용하여 기능 중요성을 비교
# model = RandomForestClassifier ()
# model.fit (X, y)
# print (model.feature_importances_)
# feat_importances = pd.Series (model.feature_importances_, index = X.columns)
# feat_importances.nlargest (8) .plot (kind = 'barh')
# plt.show ()

#기능이 서로 어떻게 관련되어 있는지 또는 대상 변수와 관련이 있는지 보여주기 위해 상관 행렬
corrmat = df.corr ()
top_corr_features = corrmat.index
plt.figure (figsize = (10,10))
g = sns.heatmap (df [top_corr_features] .corr (), annot = True , cmap = "RdYlGn")
plt.show()

