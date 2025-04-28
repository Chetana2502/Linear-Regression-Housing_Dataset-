import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
data = pd.read_csv('Housing.csv')

# Check for missing values
print(data.isnull().sum())

# Handle missing values (example: drop)
data.dropna(inplace=True)

# Encode categorical variables if needed
data = pd.get_dummies(data, drop_first=True)

X = data.drop('price', axis=1)  # Replace 'target_column' with your actual target column name
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MAE: {mae}, MSE: {mse}, R²: {r2}')

import matplotlib.pyplot as plt

# Assuming models are already trained
try:
    # Check if lengths are equal
    if len(X_test) == len(y_test) and len(y_test) == len(y_pred):
        x_values = X_test.iloc[:, 0]  # Use the first column of X_test
        plt.scatter(x_values, y_test, color='blue')  # Actual data points
        plt.plot(x_values, y_pred, color='red')  # Regression line
        plt.title('Regression Line')
        plt.xlabel('Independent Variable')
        plt.ylabel('Dependent Variable')
        plt.show()
    else:
        print("Error: Lengths of X_test, y_test, or y_pred do not match.")
except Exception as e:
    print(f"An error occurred: {e}")

print('Coefficients:', model.coef_)
