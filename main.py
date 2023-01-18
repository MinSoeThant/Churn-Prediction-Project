import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report



df = pd.read_csv(r"C:\Users\minso\OneDrive\Desktop\churn.csv", index_col='RowNumber')
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
  #print(pipe_y_pred)
  #print(y_test)
print(classification_report(y_test, pipe_y_pred))
X_test = test.loc[:, feature_cols]
y_test = test.Exited
pipe_y_pred = pipe.predict(X_test)
#print(classification_report(y_test, pipe_y_pred))

input_data = (58, 159660.80, 4, 0)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

#prediction = pipe.predict(input_data_reshaped)
#print(prediction)

#if (prediction[0] == 0):
  #print('The person will not churn')
#else:
  #print('The person will churn')

