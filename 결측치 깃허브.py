import datawig
import pandas as pd
import numpy
import pickle
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score


df=pd.read_csv('원본_nan.csv')
# df_with_missing_imputed = datawig.SimpleImputer.complete(df)
# print(df_with_missing_imputed['Insulin'])
# print(df_with_missing_imputed['SkinThickness'])
# x = df.iloc[:,:-1]
# y = df.iloc[:,-1]

#======================================================================
df_train, df_test = datawig.utils.random_split(df)

#Initialize a SimpleImputer model
imputer = datawig.SimpleImputer(
    input_columns=['BMI'], # column(s) containing information about the column we want to impute
    output_column='Outcome', # the column we'd like to impute values for
    output_path = 'imputer_model' # stores model data and metrics
    )

#Fit an imputer model on the train data
imputer.fit(train_df=df_train, num_epochs=50)

#Impute missing values and return original dataframe with predictions
imputed = imputer.predict(df_test)
print()


# with open('imputer_model/model-0025.params','rb') as f:
#     data = .load(f)
#
# print(data)