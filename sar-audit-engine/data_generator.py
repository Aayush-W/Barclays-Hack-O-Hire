import pandas as pd
import numpy as np

def generate_transaction_data():
    """
    Generate 100 sample transactions (50 normal, 50 suspicious)
    """
    np.random.seed(42)
    
    # Normal transactions
    normal = pd.DataFrame({
        'transaction_amount': np.random.uniform(1000, 50000, 50),
        'num_source_accounts': np.random.randint(1, 5, 50),
        'time_window_days': np.random.randint(30, 365, 50),
        'immediate_outflow_ratio': np.random.uniform(0, 0.3, 50),
        'customer_age_years': np.random.randint(25, 65, 50),
        'is_suspicious': 0
    })
    
    # Suspicious transactions
    suspicious = pd.DataFrame({
        'transaction_amount': np.random.uniform(100000, 5000000, 50),
        'num_source_accounts': np.random.randint(30, 100, 50),
        'time_window_days': np.random.randint(1, 14, 50),
        'immediate_outflow_ratio': np.random.uniform(0.8, 1.0, 50),
        'customer_age_years': np.random.randint(25, 65, 50),
        'is_suspicious': 1
    })
    
    # Combine
    data = pd.concat([normal, suspicious], ignore_index=True)
    
    # Shuffle
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return data

if __name__ == "__main__":
    data = generate_transaction_data()
    print(data.head(10))
    print(f"\nTotal transactions: {len(data)}")
    print(f"Suspicious: {data['is_suspicious'].sum()}")
    print(f"Normal: {(data['is_suspicious'] == 0).sum()}")