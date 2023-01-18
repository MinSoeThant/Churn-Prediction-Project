import pickle
import numpy as np
import pandas as pd

df = pd.read_csv(r"C:\Users\minso\files\churn.csv", index_col='RowNumber')
print(df)
not_churn_df = df[df['Exited'] == 0]
churn_df = df[df['Exited'] == 1]
churn_df_sample = churn_df.sample(n=1600, random_state=42)
not_churn_df_sample = not_churn_df.sample(n=6400, random_state=42)
churn_df_sample.Gender.replace(['Male', 'Female'], [0, 1], inplace=True)
churn_df_sample.Geography.replace(['France', 'Germany', 'Spain'], [0, 1, 2], inplace=True)
not_churn_df_sample.Gender.replace(['Male', 'Female'], [0, 1], inplace=True)
not_churn_df_sample.Geography.replace(['France', 'Germany', 'Spain'], [0, 1, 2], inplace=True)
train_sample = pd.concat([churn_df_sample, not_churn_df_sample])
train_sample = train_sample.drop(columns=['CustomerId', 'Surname', 'HasCrCard', 'Tenure', 'EstimatedSalary'])
feature_cols = ['Age', 'Balance', 'NumOfProducts', 'IsActiveMember']
test = df.sample(n=2000, random_state=42)
X_train = train_sample.loc[:, feature_cols]
y_train = train_sample.Exited
pipe = pickle.load(open(r'C:\Users\minso\pipe.pkl', 'rb'))
pipe.fit(X_train, y_train)
X_test = test.loc[:, feature_cols]
y_test = test.Exited
pipe_y_pred = pipe.predict(X_test)
X_test = test.loc[:, feature_cols]
y_test = test.Exited
pipe_y_pred = pipe.predict(X_test)

