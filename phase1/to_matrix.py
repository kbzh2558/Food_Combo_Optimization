from scipy.sparse import lil_matrix, save_npz
from joblib import load
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
base_dir = os.getcwd()
data = load(os.path.join(base_dir, "data_processed", "sales_record_dict.pkl"))

print("✅ Dictionary loaded successfully!")

print('====================Concat Start=========================')

print('-----step 1------')
# Step 1: Preprocess to get all unique items and transactions
all_items = set()
all_transactions = set()

for date, date_df in data.items():
    date_df = date_df.reset_index().rename(columns={'index': 'item'})
    all_items.update(date_df['item'])
    all_transactions.update(date_df.columns[date_df.columns != 'item'].tolist())

global_items = {item: idx for idx, item in enumerate(sorted(all_items))}
global_transactions = {tid: idx for idx, tid in enumerate(sorted(all_transactions))}
final_shape = (len(global_transactions), len(global_items))

del all_items, all_transactions

print('-----step 2------')

# Step 2: Build the sparse matrix incrementally
final_matrix = lil_matrix(final_shape, dtype=np.float32)

for date, date_df in tqdm(data.items()):
    date_df = date_df.reset_index().rename(columns={'index': 'item'})
    for col in date_df.columns:
        if col == 'item':
            continue
        non_zero = date_df[date_df[col] > 0]
        if not non_zero.empty:
            row_idx = global_transactions[col]
            col_indices = non_zero['item'].map(global_items).values
            final_matrix[row_idx, col_indices] = non_zero[col].values
        del non_zero

print('====================Concat End=========================')

# Save results
save_npz(os.path.join(base_dir, "data_processed", "sales_record.npz"), final_matrix.tocsr())

# Save item and transaction mappings
pd.Series(list(global_items.keys())).to_csv(os.path.join(base_dir, "data_processed", "global_items.csv"), index=False)
pd.Series(list(global_transactions.keys())).to_csv(os.path.join(base_dir, "data_processed", "global_transactions.csv"), index=False)

# combined_df.T.to_parquet(os.path.join(base_dir, "data_processed", "sales_record.parquet"), engine="pyarrow")

print(f"✅ File saved successfully at: {os.path.join(base_dir, "data_processed", "sales_record.npz")}")