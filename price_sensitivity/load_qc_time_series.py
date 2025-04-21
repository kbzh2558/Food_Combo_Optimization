import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
import dask.dataframe as dd
from dask.distributed import Client
import dask
from pyarrow.lib import ArrowMemoryError, ArrowInvalid
import re

# base directory
base_dir = "" 

def process_single_date_pandas(
    date_str: str,
    base_dir: str,
    site_list:pd.Series,
) -> pd.DataFrame:
    """
    Reads one transaction_date=YYYY-MM-DD folder for both headers and items in pure Pandas.
    Applies the same merges, filtering, and groupby logic as the Dask pipeline, but only
    for the specified date_str (e.g. "2024-02-01").

    Args:
        date_str (str): A string date in "YYYY-MM-DD" format (e.g. "2024-02-01").
        base_dir (str): Base directory where 'data' folders exist.
        products (pd.DataFrame): Preloaded dimension data for products.
        promo (pd.DataFrame): Preloaded dimension data for promotions.
        sampled_sites (np.ndarray): Array of site numbers to filter on.

    Returns:
        pd.DataFrame: A final consolidated DataFrame for the specified date,
                      including grouped lists of items, promotions, etc.
    """

  
    # ---------------------------
    # 2) Read Items (Pandas)
    # ---------------------------
    items_dir = os.path.join(base_dir, "transaction_line_items", f"transaction_date={date_str}")
    if not os.path.exists(items_dir):
        print(f"X No folder found for items: {items_dir}")
        return pd.DataFrame()

    item_files = [f for f in os.listdir(items_dir) if f.endswith(".parquet")]
    if not item_files:
        print(f"X No Parquet files in {items_dir}")
        return pd.DataFrame()

    item_dfs = []
    for f in item_files:
        file_path = os.path.join(items_dir, f)
        if os.path.getsize(file_path) == 0:
            print(f"(!) Skipping empty item file: {file_path}")
            continue
        try:
            it_df = pd.read_parquet(file_path,
                                    columns=['business_date','product_desc','quantity_sold','sales_amount','site_number','product_id'],
                                    filters = [
                                        ("site_number", "in", list(site_list))#,
                                    ])
            item_dfs.append(it_df)
            del it_df
        except ArrowMemoryError as e:
            print(e)
            try:
                ddf = dd.read_parquet(
                    file_path,
                    engine='pyarrow',
                    columns=['business_date','product_desc','quantity_sold','sales_amount','site_number','product_id'],
                    filters = [
                        ("site_number", "in", list(site_list))#,
                            ])
                
                # Compute partitions incrementally
                for partition in ddf.partitions:
                    it_df = partition.compute()
                    item_dfs.append(it_df)
                    del it_df

            except ValueError as e:
                print(e)
                try:
                    table = pq.read_table(file_path)
                    it_df = table.to_pandas()
                    item_dfs.append(it_df)
                    del it_df
                except ValueError as e:
                    print(f"Failed to load {file_path}: {e}")
                    return pd.DataFrame()  # return an empty DataFrame in case of error

    if not item_dfs:
        print(f"X All item files for {date_str} are empty or invalid.")
        return pd.DataFrame()

    transaction_item = pd.concat(item_dfs, ignore_index=True).drop_duplicates()
    transaction_item['item_id'] = transaction_item['product_id'].apply(lambda x: x.split("|")[0])
    transaction_item = transaction_item[['business_date','product_desc','item_id','site_number','quantity_sold','sales_amount']].copy()

    print('------read items done------')
    del item_dfs
    
    # Filter out negative or weird quantity
    transaction_item.quantity_sold=transaction_item.quantity_sold.astype('float')
    transaction_item = transaction_item[transaction_item["quantity_sold"] >= 0]

    # Price column
    transaction_item["price"] = (transaction_item["sales_amount"] / transaction_item["quantity_sold"]).round(2)

    print('------filter done------')

    # --------------------------------------------------
    # 5) Agg and avg the sales and price of products
    # --------------------------------------------------
    # groupby => apply(list) logic
    grouped_df = transaction_item.groupby(['business_date','item_id','product_desc']).agg({
        'quantity_sold': 'sum',
        'price': 'mean'
    }).reset_index()

    del transaction_item

    grouped_df["business_date"] = pd.to_datetime(grouped_df["business_date"], errors="coerce")
    
    # Additional date fields
    grouped_df["year"] = grouped_df["business_date"].dt.year
    grouped_df["month"] = grouped_df["business_date"].dt.month
    grouped_df["week"] = grouped_df["business_date"].dt.isocalendar().week

    print('------aggregation done------')


    print(f"Processed date {date_str}: final shape {grouped_df.shape}")
    return grouped_df.drop_duplicates(subset=['product_desc'])


# define date range
date_range=[(datetime(2021, 1, 1) + timedelta(days=i)).strftime('%Y-%m-%d') for i in range((datetime(2025, 2, 12) - datetime(2021, 1, 1)).days + 1)]

# load sites 
site_path = os.path.join(base_dir, "site")
site_files = [os.path.join(site_path, f) for f in os.listdir(site_path) if f.endswith(".parquet")]
sites = pd.concat([pd.read_parquet(f) for f in site_files], ignore_index=True)

# quebec only
sites = sites[sites['site_state_desc']=='Quebec']['site_number']

# loop through each date in the range to fetch data
data = {}

for date in tqdm(date_range, desc="Processing Dates", unit="date"):
    print(f'====================Read Start: date={date}=========================')
    # load from data
    df_single_date = process_single_date_pandas(
        date_str=date,
        base_dir=base_dir,
        site_list=list(sites)
    )
 
    # add to dict
    data[date] = df_single_date
    print(f'====================Read End: date={date}=========================')

    # delete to save memory
    del df_single_date
    
print(f'====================Saving=========================')
# save dict
from joblib import dump

# Save the large dictionary to a file
dump(data, os.path.join(base_dir, "data_processed", "qc_time_series.pkl"))

print("Dictionary saved successfully!")

# concat
stacked_df = pd.concat(data.values(), keys=data.keys()).reset_index(drop=True)
stacked_df.to_csv(os.path.join(base_dir, "data_processed", "qc_time_series.csv"), index=False)

print("CSV saved successfully!")