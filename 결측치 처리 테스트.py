from missingpy import KNNImputer

import pandas as pd

df = pd.read_csv("원본_nan.csv")
# print(df)
# imputer = KNNImputer(n_neighbors=2,weights="uniform") ##최고성능
imputer = KNNImputer(n_neighbors=2,weights="uniform",col_max_missing=0.9,row_max_missing=0.9)
X_imputed = imputer.fit_transform(df)
print(X_imputed)

# df2=pd.DataFrame(X_imputed, columns=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome'])
# print(df2['Insulin'])
# df2.to_csv('test_imputed.csv', index=False, encoding='cp949', mode='w')

#https://github.com/epsilon-machine/missingpy