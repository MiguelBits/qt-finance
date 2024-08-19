import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('data.csv')

# Convert the 'Day' column to datetime
df['Day'] = pd.to_datetime(df['Day'])

# Create a numeric representation of dates (number of Days since the first date)
df['Days_since_start'] = (df['Day'] - df['Day'].min()).dt.days

# Prepare the data for linear regression
X = df['Days_since_start'].values.reshape(-1, 1)
y = df['mvrv'].values

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Plot the results
plt.figure(figsize=(12, 6))
plt.scatter(df['Day'], df['mvrv'], color='blue', label='Actual MVRV')
plt.plot(df['Day'], y_pred, color='red', label='Linear Regression')
plt.title('MVRV Linear Regression')
plt.xlabel('Date')
plt.ylabel('MVRV')
plt.legend()
plt.grid(True)
plt.show()

# Print the model coefficients
print(f"Intercept: {model.intercept_}")
print(f"Slope: {model.coef_[0]}")

# Calculate R-squared
r_squared = model.score(X, y)
print(f"R-squared: {r_squared}")