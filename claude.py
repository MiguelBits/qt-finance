import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta

# Fetch Bitcoin data
end_date = datetime.now()
start_date = end_date - timedelta(days=365)  # Last year of data

btc_data = yf.download('BTC-USD', start=start_date, end=end_date)

# Display the first few rows of the data
print(btc_data.head())

# Plot closing price
plt.figure(figsize=(12, 6))
plt.plot(btc_data.index, btc_data['Close'])
plt.title('Bitcoin Closing Price - Last Year')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.grid(True)
plt.show()

# Calculate daily returns
btc_data['Returns'] = btc_data['Close'].pct_change()

# Display summary statistics
print(btc_data['Returns'].describe())

# Plot a histogram of returns
plt.figure(figsize=(10, 6))
btc_data['Returns'].hist(bins=50)
plt.title('Distribution of Daily Returns')
plt.xlabel('Daily Returns')
plt.ylabel('Frequency')
plt.show()