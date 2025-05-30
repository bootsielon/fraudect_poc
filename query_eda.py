import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Fix suffixes and timestamp conversion issue
# Reload and merge tables correctly
conn = sqlite3.connect("fraud_poc.db")

df_clients = pd.read_sql_query("SELECT * FROM clients", conn)
df_merchants = pd.read_sql_query("SELECT * FROM merchants", conn)
df_transactions = pd.read_sql_query("SELECT * FROM transactions", conn)

# Fix suffixes explicitly
df_clients.rename(columns={"account_creation_date": "account_creation_date_client"}, inplace=True)
df_merchants.rename(columns={"account_creation_date": "account_creation_date_merchant"}, inplace=True)

# Merge
df = df_transactions.merge(df_clients, on="account_id")
df = df.merge(df_merchants, on="merchant_id")

# Convert timestamps
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["account_creation_date_client"] = pd.to_datetime(df["account_creation_date_client"])
df["account_creation_date_merchant"] = pd.to_datetime(df["account_creation_date_merchant"])

# Create derived features
df["transaction_hour"] = df["timestamp"].dt.hour
df["transaction_day"] = df["timestamp"].dt.dayofweek
df["client_account_age_days"] = (df["timestamp"] - df["account_creation_date_client"]).dt.days
df["merchant_account_age_days"] = (df["timestamp"] - df["account_creation_date_merchant"]).dt.days

# Summary statistics
summary_stats = df.describe(include='all')

# Plots
plt.figure(figsize=(6, 4))
sns.countplot(x="is_fraud", data=df)
plt.title("Fraud Class Distribution")
plt.savefig("/mnt/data/fraud_class_distribution.png")
plt.close()

plt.figure(figsize=(10, 4))
sns.histplot(df["amount"], bins=50, kde=True)
plt.title("Transaction Amount Distribution")
plt.savefig("/mnt/data/amount_distribution.png")
plt.close()

plt.figure(figsize=(10, 4))
sns.boxplot(x="is_fraud", y="amount", data=df)
plt.title("Amount by Fraud Label")
plt.savefig("/mnt/data/amount_by_fraud.png")
plt.close()

plt.figure(figsize=(10, 4))
sns.histplot(df["client_account_age_days"], bins=50)
plt.title("Client Account Age Distribution")
plt.savefig("/mnt/data/client_account_age_distribution.png")
plt.close()

plt.figure(figsize=(10, 4))
sns.histplot(df["merchant_account_age_days"], bins=50)
plt.title("Merchant Account Age Distribution")
plt.savefig("/mnt/data/merchant_account_age_distribution.png")
plt.close()

# import ace_tools as tools; tools.display_dataframe_to_user(name="Fraud Dataset Summary Statistics", dataframe=summary_stats)
