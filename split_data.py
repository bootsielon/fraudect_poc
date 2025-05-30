import os
import pandas as pd

# Define the source directory
src_dir = r"C:\Users\i_bau\OneDrive\aaaWork\repos\frauprev\data\kagglebankfraud"

# Iterate over all CSV files starting with 'Variant'
for filename in os.listdir(src_dir):
    if filename.startswith("Variant") and filename.endswith(".csv"):
        file_path = os.path.join(src_dir, filename)
        print(f"Processing: {filename}")

        # Load the CSV
        df = pd.read_csv(file_path)

        # Determine midpoint
        mid = len(df) // 2

        # Split the DataFrame
        part1 = df.iloc[:mid]
        part2 = df.iloc[mid:]

        # Write both parts
        base = os.path.splitext(filename)[0]
        part1.to_csv(os.path.join(src_dir, f"{base}_part1.csv"), index=False)
        part2.to_csv(os.path.join(src_dir, f"{base}_part2.csv"), index=False)

        print(f"Saved: {base}_part1.csv and {base}_part2.csv")
