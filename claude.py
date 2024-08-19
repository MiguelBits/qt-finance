import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats

# Fetch Bitcoin price data
btc = yf.Ticker("BTC-USD")
btc_data = btc.history(start="2016-01-01", end=datetime.now().strftime('%Y-%m-%d'))

# Calculate a 365-day moving average as a proxy for "realized value"
btc_data['MA365'] = btc_data['Close'].rolling(window=365).mean()

# Calculate our MVRV proxy (current price / 365-day MA)
btc_data['MVRV_proxy'] = btc_data['Close'] / btc_data['MA365']

# Drop NaN values (first year will have NaNs due to the moving average)
btc_data.dropna(inplace=True)

# Prepare data for linear regression
X = btc_data.index.astype(int).values.reshape(-1, 1)
y = btc_data['MVRV_proxy'].values.reshape(-1, 1)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform linear regression
model = LinearRegression()
model.fit(X_scaled, y)

# Generate predictions
y_pred = model.predict(X_scaled)

# Calculate residuals
residuals = y - y_pred

# Calculate z-score of residuals
z_score_residuals = stats.zscore(residuals)

# Add z-score to the dataframe
btc_data['MVRV_z_score'] = z_score_residuals

# Create the plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 16), sharex=True)

# Plot Bitcoin price
ax1.set_ylabel('Bitcoin Price (USD)', color='tab:blue')
ax1.plot(btc_data.index, btc_data['Close'], color='tab:blue', label='Bitcoin Price')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Plot MVRV z-score
ax2.set_xlabel('Date')
ax2.set_ylabel('MVRV Z-Score', color='tab:green')
ax2.plot(btc_data.index, btc_data['MVRV_z_score'], color='tab:green', label='MVRV Z-Score')
ax2.tick_params(axis='y', labelcolor='tab:green')

# Add horizontal lines at z-score = -3, -2, -1, 0, 1, 2, 3
for i in range(-3, 4):
    color = 'red' if abs(i) == 3 else 'orange' if abs(i) == 2 else 'yellow' if abs(i) == 1 else 'gray'
    ax2.axhline(y=i, color=color, linestyle='--', alpha=0.7)

# Set title and display the plot
plt.suptitle('Bitcoin Price and MVRV Z-Score based on Linear Regression (2016 - Present)')
fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.98))
fig.tight_layout()
plt.show()

# Print some statistics
print("Bitcoin Price Statistics:")
print(btc_data['Close'].describe())
print("\nMVRV Z-Score Statistics:")
print(btc_data['MVRV_z_score'].describe())

# Print linear regression details
print("\nLinear Regression Details:")
print(f"Slope: {model.coef_[0][0]}")
print(f"Intercept: {model.intercept_[0]}")
print(f"R-squared: {model.score(X_scaled, y)}")

# Save the data to a CSV file
btc_data['MVRV_regression'] = y_pred
btc_data.to_csv('bitcoin_with_mvrv_zscore_regression.csv')
print("\nData saved to 'bitcoin_with_mvrv_zscore_regression.csv'")