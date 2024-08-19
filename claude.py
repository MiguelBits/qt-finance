import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime

# Fetch Bitcoin price data
btc = yf.Ticker("BTC-USD")
btc_data = btc.history(start="2016-01-01", end=datetime.now().strftime('%Y-%m-%d'))

# Calculate a 365-day moving average as a proxy for "realized value"
btc_data['MA365'] = btc_data['Close'].rolling(window=365).mean()

# Calculate our MVRV proxy (current price / 365-day MA)
btc_data['MVRV_proxy'] = btc_data['Close'] / btc_data['MA365']

# Drop NaN values (first year will have NaNs due to the moving average)
btc_data.dropna(inplace=True)

# Create the plot
fig, ax1 = plt.subplots(figsize=(16, 8))

# Plot Bitcoin price
ax1.set_xlabel('Date')
ax1.set_ylabel('Bitcoin Price (USD)', color='tab:blue')
ax1.plot(btc_data.index, btc_data['Close'], color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Create a second y-axis for MVRV proxy
ax2 = ax1.twinx()
ax2.set_ylabel('MVRV Proxy', color='tab:orange')
ax2.plot(btc_data.index, btc_data['MVRV_proxy'], color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:orange')

# Add horizontal lines at MVRV proxy = 1, 2, and 3
ax2.axhline(y=1, color='green', linestyle='--', alpha=0.7)
ax2.axhline(y=2, color='yellow', linestyle='--', alpha=0.7)
ax2.axhline(y=3, color='red', linestyle='--', alpha=0.7)

# Set title and display the plot
plt.title('Bitcoin Price and MVRV Proxy (2016 - Present)')
fig.tight_layout()
plt.show()

# Print some statistics
print("Bitcoin Price Statistics:")
print(btc_data['Close'].describe())
print("\nMVRV Proxy Statistics:")
print(btc_data['MVRV_proxy'].describe())

# Save the data to a CSV file
btc_data.to_csv('bitcoin_with_mvrv_proxy.csv')
print("\nData saved to 'bitcoin_with_mvrv_proxy.csv'")