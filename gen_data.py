import sqlite3
import pandas as pd
import numpy as np
import random
from faker import Faker
from datetime import datetime, timedelta

# Initialize Faker
# fake = Faker()

# --- 1. DATABASE SETUP ---

def create_tables(cursor):
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS clients (
        account_id TEXT PRIMARY KEY,
        account_creation_date TEXT,
        avg_spend REAL,
        country TEXT,
        ip_address TEXT,
        device_id TEXT,
        has_biometrics INTEGER
    )
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS merchants (
        merchant_id TEXT PRIMARY KEY,
        name TEXT,
        category TEXT,
        risk_score REAL,
        account_creation_date TEXT,
        country TEXT
    )
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS transactions (
        transaction_id TEXT PRIMARY KEY,
        timestamp TEXT,
        amount REAL,
        currency TEXT,
        location TEXT,
        ip_address TEXT,
        device_id TEXT,
        account_id TEXT,
        merchant_id TEXT,
        is_fraud INTEGER,
        FOREIGN KEY(account_id) REFERENCES clients(account_id),
        FOREIGN KEY(merchant_id) REFERENCES merchants(merchant_id)
    )
    ''')

# --- 2. DATA GENERATORS ---

def generate_clients(fake, cd=100, random_seed=42):
    random.seed(random_seed)
    return [
        (
            fake.uuid4(),
            fake.date_between(start_date='-5y', end_date='-1d').isoformat(),
            round(random.uniform(20, 1000), 2),
            fake.country(),
            fake.ipv4(),
            fake.uuid4(),
            random.choice([0, 1])
        )
        for _ in range(cd)
    ]

def generate_merchants(fake, m=50, seed=42):
    random.seed(seed)
    return [
        (
            fake.uuid4(),
            fake.company(),
            random.choice(['Retail', 'Electronics', 'Food', 'Fashion', 'Travel', 'Gaming']),
            round(random.uniform(0, 1), 2),
            fake.date_between(start_date='-10y', end_date='-1d').isoformat(),
            fake.country()
        )
        for _ in range(m)
    ]

def generate_transactions(fake, clients, merchants, n=10000):
    return [
        (
            fake.uuid4(),
            (datetime.now() - timedelta(minutes=random.randint(0, 1000000))).isoformat(),
            round(random.uniform(1, 5000), 2),
            random.choice(['USD', 'EUR', 'GBP', 'MXN']),
            fake.city(),
            fake.ipv4(),
            fake.uuid4(),
            random.choice(clients)[0],
            random.choice(merchants)[0],
            int(np.random.choice([0, 1], p=[0.95, 0.05]))  # Ensure is_fraud is an integer
        )
        for _ in range(n)
    ]

# --- 3. MAIN EXECUTION ---

def gen_data(n=1000, random_seed=42):
    """
    Generate synthetic data and store it in a SQLite database.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    fake = Faker()
    Faker.seed(random_seed) # Seed Faker for reproducibility
    # Connect to SQLite (use ':memory:' for RAM or 'fraud_poc.db' to persist)
    conn = sqlite3.connect('fraud_poc.db')
    cursor = conn.cursor()
    # Create tables
    create_tables(cursor)
    # Generate and insert synthetic data
    client_data = generate_clients(fake, cd=50, random_seed=random_seed)
    # Generate merchants and transactions
    merchant_data = generate_merchants(fake, m=50, seed=random_seed)
    transaction_data = generate_transactions(fake, client_data, merchant_data, n=10000)
    cursor.executemany('INSERT INTO clients VALUES (?, ?, ?, ?, ?, ?, ?)', client_data)
    cursor.executemany('INSERT INTO merchants VALUES (?, ?, ?, ?, ?, ?)', merchant_data)
    cursor.executemany('INSERT INTO transactions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', transaction_data)
    conn.commit()
    # Optional: shown sample
    sample_df = pd.read_sql_query("SELECT * FROM transactions LIMIT 5", conn)
    print(sample_df)
    conn.close()


def generate_dataframes(n_clients=50, n_merchants=50, n_transactions=10000, random_seed=42):
    fake = Faker()
    Faker.seed(random_seed)
    clients = generate_clients(fake, cd=n_clients, random_seed=random_seed)
    merchants = generate_merchants(fake, m=n_merchants, seed=random_seed)
    transactions = generate_transactions(fake, clients, merchants, n=n_transactions)

    df_clients = pd.DataFrame(clients, columns=[
        "account_id", "account_creation_date", "avg_spend", "country", "ip_address", "device_id", "has_biometrics"
    ])
    df_merchants = pd.DataFrame(merchants, columns=[
        "merchant_id", "name", "category", "risk_score", "account_creation_date", "country"
    ])
    df_transactions = pd.DataFrame(transactions, columns=[
        "transaction_id", "timestamp", "amount", "currency", "location", "ip_address", "device_id",
        "account_id", "merchant_id", "is_fraud"
    ])
    return df_clients, df_merchants, df_transactions


if __name__ == "__main__":
    gen_data(n=10000, random_seed=4024)
    # Uncomment the line below to generate data with a different seed
    # gen_data(n=1000, random_seed=24)

