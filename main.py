#import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import joblib

#load your dataset
file_path = 'House_details.csv'
data = pd.read_csv(file_path)

def preprocess_data(data, target_column):
 categorical_columns = data.select_dtypes(include=['object']).columns
 data_encoded = pd.get_dummies(data, columns=categorical_columns)

 #Define features X and target variable y
 X = data_encoded.drop(columns=target_column)
 y = data_encoded[target_column]

 return X, y

def train_and_evaluate(X_train, X_test, y_train, y_test):
 #initialise the linear regression model
 model = LinearRegression()

 #Train the model
 model.fit(X_train, y_train)

 #make predictions on the test set
 predictions = model.predict(X_test)

 #evaluate the performance of the model
 mse = mean_squared_error(y_test, predictions)
 print(f'Mean Squared Error: {mse}')

 #save the trained model to a file
 joblib.dump(model, 'linear_regression_model.joblib')

def predict_price(input_features):
 model = joblib.load('linear_regression_model.joblib')

 prediction = model.predict([input_features])

 return prediction[0]

if __name__=="__main__":

 #preprocess data
 target_column = 'Price'
 X, y = preprocess_data(data, target_column)

 #split the data into training and testing sets
 X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42)

 #train and evaluate the model
 train_and_evaluate(X_train, X_test, y_train, y_test)

 #Example: Predict the price using the trained model
 input_features =[1000, 3, 'delhi',...]

 #Replace the actual values
 predicted_price = predict_price(input_features)
 print(f'Predicted Price: {predicted_price}')

