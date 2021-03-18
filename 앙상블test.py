import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
pd.set_option('display.expand_frame_repr', False)

data_df=pd.read_csv('/home/hong/Downloads/diabetes.csv')
# data_df = pd.read_csv("test_imputed.csv")

lr_clf = LogisticRegression()
knn_clf=KNeighborsClassifier(n_neighbors=8)

vo_clf = VotingClassifier(estimators=[('LR',lr_clf),('KNN',knn_clf)], voting='soft')

col = list(map(str, data_df.columns))
x=data_df[col[:-1]]
y=data_df[col[-1]]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, shuffle=True)

vo_clf.fit(x_train,y_train)
pred = vo_clf.predict(x_test)
print('Voting acc: {0:.4f}'.format(accuracy_score(y_test, pred)))

classifiers = [lr_clf, knn_clf]
for classifiers in classifiers :
    classifiers.fit(x_train,y_train)
    pred=classifiers.predict(x_test)
    Class_name = classifiers.__class__.__name__
    print('{0} acc: {1:.4f}'.format(Class_name,accuracy_score(y_test,pred)))