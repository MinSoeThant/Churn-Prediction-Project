import numpy as np
import streamlit as st
import pickle
from main import X_train, y_train, X_test, y_test

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
