# Run code cell in Sublime : Import the necessary Python modules and create the 'prediction()' function as directed above.
# Importing the necessary Python modules.
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error
# Define the 'prediction()' function.
@st.cache()
def prediction(car_df, car_width, engine_size, horse_power, drive_wheel_fwd, car_comp_buick):
    X = car_df.iloc[:, :-1] 
    y = car_df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42) 

    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    score = lin_reg.score(X_train, y_train)

    price = lin_reg.predict([[car_width, engine_size, horse_power, drive_wheel_fwd, car_comp_buick]])
    price = price[0]

    y_test_pred = lin_reg.predict(X_test)
    test_r2_score = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_msle = mean_squared_log_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    return price, score, test_r2_score, test_mae, test_msle, test_rmse
def app(car_df):
    st.markdown("<p style='color:blue;font-size:25px;'>This app uses <b>Linear regression</b> to predict the price of a car based on your inputs.</p>", unsafe_allow_html = True)
    st.subheader('Select values:')
    wd = st.slider('Car Width', float(car_df['carwidth'].min()), float(car_df['carwidth'].max()))
    es = st.slider('Engine Size', float(car_df['enginesize'].min()), float(car_df['enginesize'].max()))
    hp = st.slider('Horse Power', float(car_df['horsepower'].min()), float(car_df['horsepower'].max()))
    fwd = st.radio('Is it a forward drive wheel car?', ('Yes', 'No'))
    if fwd == 'No':
        fwd = 0
    else:
        fwd = 1
    bk = st.radio('Is the car manifactured by Buick?', ('Yes', 'No'))
    if bk == 'No':
        bk = 0
    else:
        bk = 1
    if st.button('Predict'):
        st.subheader('Prediction Results:')
        price, score, r2, mae, msle, rmse = prediction(car_df, wd, es, hp, fwd, bk)
        st.success(f'The price of the car is ₹ {round(price*78.14, 2)}.')
        st.info(f'Accuracy of the model by which the price was predicted was {round(score*100,2)}%.')
        st.info(f'Accuracy of the model by which the price was predicted was {round(score*100,2)}%.')
        st.info(f'R2 Score of this model is {r2}.')
        st.info(f'MAE of this model is {mae}.')
        st.info(f'MSLE of this model is {msle}.')
        st.info(f'RMSE of this model is {rmse}.')