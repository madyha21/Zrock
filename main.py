#import necessary libraries
import pandas as pd
import numpy as np
import streamlit as st

#loading the data
data = pd.read_csv("C:\\Users\\madih\\OneDrive\\Desktop\\\Zrock\House_details.csv")

#Define features X and target variable y
X = data[['area','num_rooms','location','year_built']]
y = data['Price']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

ohe = OneHotEncoder()
ohe.fit(X[['area','num_rooms','location','year_built']])

column_trans = make_column_transformer((OneHotEncoder
                                        (handle_unknown='ignore'),['area','num_rooms','location','year_built']),
                                        remainder='passthrough')

lr = LinearRegression()
pipe = make_pipeline(column_trans,lr)
pipeline = make_pipeline(column_trans,lr)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
r2_score(y_test, y_pred)

import pickle
pickle.dump(pipe,open('LinearRegressionModel.pkl','wb'))
pipe.predict(pd.DataFrame(columns=['area','num_rooms','location','year_built'],
                          data = np.array([2000,6,'AP',2001]).reshape(1,4)))
pipe.steps[0][1].transformers[0][1].categories[0]

#streamlit app
st.title('House Price Prediction System')

#sidebar for using input
area = st.selectbox('select the area:', data['area'].unique())
num_rooms = st.selectbox('select the num_rooms:', data['num_rooms'].unique())
location = st.selectbox('select the area:', data['location'].unique())
year_built = st.selectbox('select the year_built:', data['year_built'].unique())
predicted_price = pipeline.predict(data)[0]

#Button to trigger prediction
if st.button('Predict Price'):
    predicted_price = pipeline.predict(data)[0]


#display the prediction
st.write(f'Predicted Price: {predicted_price}')
