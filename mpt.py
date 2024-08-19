import pandas as pd
import numpy as np
from pycoingecko import CoinGeckoAPI
from scipy import stats
import time

def fetch_coin_data(cg, coin_id, days=365):
    try:
        data = cg.get_coin_market_chart_by_id(id=coin_id, vs_currency='usd', days=days)
        df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df['returns'] = df['price'].pct_change()
        return df
    except Exception as e:
        print(f"Error fetching data for {coin_id}: {e}")
        return None

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    excess_returns = returns - risk_free_rate / 365
    return np.sqrt(365) * excess_returns.mean() / excess_returns.std()

def calculate_omega_ratio(returns, threshold=0):
    returns_above_threshold = returns[returns > threshold]
    returns_below_threshold = returns[returns <= threshold]
    
    if len(returns_below_threshold) == 0:
        return np.inf
    
    return returns_above_threshold.sum() / abs(returns_below_threshold.sum())

cg = CoinGeckoAPI()

# Fetch top 100 coins by market cap
top_coins = cg.get_coins_markets(vs_currency='usd', order='market_cap_desc', per_page=100, page=1)

results = []

for coin in top_coins:
    print(f"Processing {coin['name']}...")
    df = fetch_coin_data(cg, coin['id'])
    
    if df is not None and len(df) > 0:
        sharpe_ratio = calculate_sharpe_ratio(df['returns'].dropna())
        omega_ratio = calculate_omega_ratio(df['returns'].dropna())
        
        results.append({
            'Coin': coin['name'],
            'Symbol': coin['symbol'].upper(),
            'Market Cap Rank': coin['market_cap_rank'],
            'Sharpe Ratio': sharpe_ratio,
            'Omega Ratio': omega_ratio
        })
    
    time.sleep(1)  # To avoid hitting API rate limits

# Create DataFrame from results
results_df = pd.DataFrame(results)

# Sort by Sharpe Ratio descending
results_df = results_df.sort_values('Sharpe Ratio', ascending=False)

# Display results
pd.set_option('display.max_rows', None)
print(results_df)

# Save to CSV
results_df.to_csv('crypto_ratios.csv', index=False)
print("\nResults saved to 'crypto_ratios.csv'")