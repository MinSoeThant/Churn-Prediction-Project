import numpy as np
import streamlit as st
import pickle
import pandas as pd


df = pd.read_csv(r"C:\Users\minso\files\churn.csv", index_col='RowNumber')
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

X_test = test.loc[:, feature_cols]
y_test = test.Exited




pipe = pickle.load(open(r'C:\Users\minso\pipe.pkl', 'rb'))
pipe.fit(X_train, y_train)
pipe_y_pred = pipe.predict(X_test)





def churn_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = pipe.predict(input_data_reshaped)
    print(prediction)

    if prediction[0] == 0:
        return 'This person will not churn'
    else:
        return 'This person will churn'


def main():
    st.title('Churn Prediction')
    Age = st.text_input('Age')
    Balance = st.text_input('Balance')
    NumOfProducts = st.text_input('Number Of Products')
    IsActiveMember = st.text_input('Active membership status (0 = No, 1 = Yes)')

    churn = ''

    if st.button('Predict'):
        churn = churn_prediction([Age, Balance, NumOfProducts, IsActiveMember])

        st.success(churn)


if __name__ == '__main__':
    main()
