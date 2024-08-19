import pandas as pd
import numpy as np
from pycoingecko import CoinGeckoAPI
import time
from datetime import datetime, timedelta

def fetch_coin_data(cg, coin_id, days=90):
    try:
        data = cg.get_coin_market_chart_by_id(id=coin_id, vs_currency='usd', days=days)
        df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        print(f"Error fetching data for {coin_id}: {e}")
        return None

def calculate_returns(df, period):
    return df['price'].pct_change(period)

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    excess_returns = returns - risk_free_rate / 365  # Daily risk-free rate
    return np.sqrt(365) * excess_returns.mean() / excess_returns.std()

def calculate_omega_ratio(returns, threshold=0):
    returns_above_threshold = returns[returns > threshold]
    returns_below_threshold = returns[returns <= threshold]
    
    if len(returns_below_threshold) == 0:
        return np.inf
    
    return returns_above_threshold.sum() / abs(returns_below_threshold.sum())

cg = CoinGeckoAPI()

# Fetch top 10 coins by market cap
top_coins = cg.get_coins_markets(vs_currency='usd', order='market_cap_desc', per_page=10, page=1)

results = []

for coin in top_coins:
    print(f"Processing {coin['name']}...")
    df = fetch_coin_data(cg, coin['id'])
    
    if df is not None and len(df) > 0:
        coin_results = {
            'Coin': coin['name'],
            'Symbol': coin['symbol'].upper(),
            'Market Cap Rank': coin['market_cap_rank']
        }
        
        # Calculate returns for different periods
        periods = {
            '1D': 1,
            '7D': 7,
            '30D': 30,
            '90D': 90
        }
        
        for period_name, days in periods.items():
            returns = calculate_returns(df, days)
            sharpe_ratio = calculate_sharpe_ratio(returns.dropna())
            omega_ratio = calculate_omega_ratio(returns.dropna())
            
            coin_results[f'Sharpe Ratio {period_name}'] = sharpe_ratio
            coin_results[f'Omega Ratio {period_name}'] = omega_ratio
        
        results.append(coin_results)
    
    time.sleep(1)  # To avoid hitting API rate limits

# Create DataFrame from results
results_df = pd.DataFrame(results)

# Function to get top 3 columns
def get_top_3_columns(df, metric):
    columns = [col for col in df.columns if col.startswith(metric)]
    return df[columns].mean().nlargest(3).index.tolist()

# Get top 3 Sharpe and Omega columns
top_sharpe_columns = get_top_3_columns(results_df, 'Sharpe Ratio')
top_omega_columns = get_top_3_columns(results_df, 'Omega Ratio')

# Combine top columns
top_columns = ['Coin', 'Symbol', 'Market Cap Rank'] + top_sharpe_columns + top_omega_columns

# Remove duplicates while preserving order
top_columns = list(dict.fromkeys(top_columns))

# Sort by the first Sharpe Ratio column in top_sharpe_columns
results_df_sorted = results_df.sort_values(top_sharpe_columns[0], ascending=False)

# Display results with top columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(results_df_sorted[top_columns])

# Save to CSV
results_df.to_csv('crypto_top10_ratios_past_90days.csv', index=False)
print("\nFull results saved to 'crypto_top10_ratios_past_90days.csv'")