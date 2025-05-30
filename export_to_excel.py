import pandas as pd
from gen_data import generate_dataframes

def export_synthetic_to_excel(out_path="synthetic_data.xlsx", n_clients=50, n_merchants=50, n_transactions=10000, random_seed=42):
    """
    Export synthetic data to an Excel file.
    """
    df_clients, df_merchants, df_transactions = generate_dataframes(
        n_clients=n_clients,
        n_merchants=n_merchants,
        n_transactions=n_transactions,
        random_seed=random_seed
    )

    with pd.ExcelWriter(out_path) as writer:
        df_clients.to_excel(writer, sheet_name="clients", index=False)
        df_merchants.to_excel(writer, sheet_name="merchants", index=False)
        df_transactions.to_excel(writer, sheet_name="transactions", index=False)

    print(f"Data saved to {out_path}")

if __name__ == "__main__":
    export_synthetic_to_excel()