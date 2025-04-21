#%%#
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
#%%#


def process_single_date_pandas(
    date_str: str,
    base_dir: str,
    products: pd.DataFrame,
    promo: pd.DataFrame,
    sampled_sites: np.ndarray
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
    # 1) Read Headers (Pandas)
    # ---------------------------
    headers_dir = os.path.join(base_dir, "data", "transaction_line_headers", f"transaction_date={date_str}")
    if not os.path.exists(headers_dir):
        print(f"❌ No folder found for headers: {headers_dir}")
        return pd.DataFrame()  # Return empty if folder doesn't exist

    header_files = [f for f in os.listdir(headers_dir) if f.endswith(".parquet")]
    if not header_files:
        print(f"❌ No Parquet files in {headers_dir}")
        return pd.DataFrame()

    header_dfs = []
    for f in header_files:
        file_path = os.path.join(headers_dir, f)
        if os.path.getsize(file_path) == 0:
            print(f"⚠️ Skipping empty header file: {file_path}")
            continue
        
        try:
            hdr_df = pd.read_parquet(file_path)
            header_dfs.append(hdr_df)
            del hdr_df
        except ArrowMemoryError as e:
            print(e)
            try:
                with dask.config.set(scheduler='threads'):
                    ddf = dd.read_parquet(file_path, engine='pyarrow', ignore_metadata_file=True, use_legacy_dataset=False)
                    
                    # sample down
                    ddf = ddf[ddf["site_number"].isin(sampled_sites)]

                    # Compute the DataFrame
                    hdr_df = ddf.compute()
                    header_dfs.append(hdr_df)
                    del hdr_df

            except ValueError as e:
                print(e)
                try:
                    table = pq.read_table(file_path)
                    hdr_df = table.to_pandas()
                    header_dfs.append(hdr_df)
                    del hdr_df
                except ValueError as e:
                    print(f"Failed to load {file_path}: {e}")
                    return pd.DataFrame()  # return an empty DataFrame in case of error
        
    if not header_dfs:
        print(f"❌ All header files for {date_str} are empty or invalid.")
        return pd.DataFrame()

    transaction_header = pd.concat(header_dfs, ignore_index=True).drop_duplicates()
    del header_dfs
    print('------read headers done------')
    
    # ---------------------------
    # 2) Read Items (Pandas)
    # ---------------------------
    items_dir = os.path.join(base_dir, "data", "transaction_line_items", f"transaction_date={date_str}")
    if not os.path.exists(items_dir):
        print(f"❌ No folder found for items: {items_dir}")
        return pd.DataFrame()

    item_files = [f for f in os.listdir(items_dir) if f.endswith(".parquet")]
    if not item_files:
        print(f"❌ No Parquet files in {items_dir}")
        return pd.DataFrame()

    item_dfs = []
    for f in item_files:
        file_path = os.path.join(items_dir, f)
        if os.path.getsize(file_path) == 0:
            print(f"⚠️ Skipping empty item file: {file_path}")
            continue
        try:
            it_df = pd.read_parquet(file_path)
            item_dfs.append(it_df)
            del it_df
        except ArrowMemoryError as e:
            print(e)
            try:
                ddf = dd.read_parquet(
                    file_path,
                    engine='pyarrow',
                    columns=["transaction_uid","site_number", "product_key", "promotion_key","business_date","quantity_sold","sales_amount"],
                    filters=[("site_number", "in", sampled_sites.tolist())]
                )
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
        print(f"❌ All item files for {date_str} are empty or invalid.")
        return pd.DataFrame()

    transaction_item = pd.concat(item_dfs, ignore_index=True).drop_duplicates()
    print('------read items done------')
    del item_dfs
    
    # --------------------------------------------------
    # 3) Merge Headers + Items (Pandas)
    # --------------------------------------------------
    df_consol = pd.merge(transaction_header, transaction_item, on="transaction_uid", how="left")

    del transaction_header,transaction_item

    # Rename columns to avoid x/y suffixes
    df_consol.rename(
        columns={
            "site_number_x": "site_number",
            "business_date_x": "business_date"
        },
        inplace=True
    )
    
    # # Sub-sample by site_number
    df_consol = df_consol[df_consol["site_number"].isin(sampled_sites)]

    # --------------------------------------------------
    # 4) Merge with dimension data (products, promo)
    # --------------------------------------------------
    # Merge with products
    df_consol = df_consol.merge(products, how="left", on="product_key")
    # Merge with promos
    df_consol = df_consol.merge(
        promo[["promotion_key", "promotion_description", "promotion_type"]],
        how="left",
        on="promotion_key"
    )

    # Filter out negative or weird quantity
    df_consol.quantity_sold=df_consol.quantity_sold.astype('float')
    df_consol = df_consol[df_consol["quantity_sold"] >= 0]

    # Price column
    df_consol["price"] = (df_consol["sales_amount"] / df_consol["quantity_sold"]).round(2)

    # Filter out categories
    exclude_cats = ["COUPONS", "Unknown", "CARTES-CADEAUX-C", "DEPOT BOUTEILLE", "FOURNITURES-SACS"]
    df_consol = df_consol[~df_consol["category_desc"].isin(exclude_cats)]

    # Drop unwanted 'sys_' columns from the merges
    columns_to_drop = [
        "sys_environment_name_x", "sys_integration_id_x", "sys_exported_by_x",
        "site_number_y", "business_date_y", "site_key_y",
        "sys_environment_name_y", "sys_integration_id_y",
        "sys_exported_dt", "sys_exported_by_y"
    ]
    df_consol.drop(columns=[c for c in columns_to_drop if c in df_consol.columns], 
                   inplace=True, errors="ignore")

    print('------merge done------')

    # --------------------------------------------------
    # 5) Group the columns into lists
    # --------------------------------------------------
    # groupby => apply(list) logic
    df_items=df_consol.groupby('transaction_uid')['item_desc'].apply(list)
    df_qty = df_consol.groupby("transaction_uid")["quantity_sold"].apply(list)
    df_price = df_consol.groupby("transaction_uid")["price"].apply(list)
    df_sub_cat = df_consol.groupby("transaction_uid")["sub_category_desc"].apply(list)
    df_dept = df_consol.groupby("transaction_uid")["category_desc"].apply(list)
    df_promo_desc = df_consol.groupby("transaction_uid")["promotion_description"].apply(list)
    df_promo_type = df_consol.groupby("transaction_uid")["promotion_type"].apply(list)

    # Prepare final minimal columns for merging
    df_result = df_consol[["transaction_uid", "site_number", "business_date"]].copy()
    df_result["business_date"] = pd.to_datetime(df_result["business_date"], errors="coerce")

    print('------grouping done------')


    # Merge back the grouped columns
    df_result = pd.merge(df_result, df_items, on="transaction_uid", how="left")
    df_result = pd.merge(df_result, df_qty, on="transaction_uid", how="left")
    df_result = pd.merge(df_result, df_price, on="transaction_uid", how="left")
    df_result = pd.merge(df_result, df_sub_cat, on="transaction_uid", how="left")
    df_result = pd.merge(df_result, df_dept, on="transaction_uid", how="left")
    df_result = pd.merge(df_result, df_promo_desc, on="transaction_uid", how="left")
    df_result = pd.merge(df_result, df_promo_type, on="transaction_uid", how="left")

    del df_qty,df_price,df_sub_cat,df_dept,df_items,df_consol


    # Additional date fields
    df_result["year"] = df_result["business_date"].dt.year
    df_result["month"] = df_result["business_date"].dt.month
    df_result["week"] = df_result["business_date"].dt.isocalendar().week

    # Summation of quantity sold
    df_result["sales_vol"] = df_result["quantity_sold"].apply(lambda x: sum(x) if isinstance(x, list) else 0)

    print(f"✅ Processed date {date_str}: final shape {df_result.shape}")

    return df_result.drop_duplicates(subset=['transaction_uid'])

# base directory
base_dir = os.getcwd()

# load products
product_path = os.path.join(base_dir, "data", "product")
product_files = [os.path.join(product_path, f) for f in os.listdir(product_path) if f.endswith(".parquet")]
products = pd.concat([pd.read_parquet(f) for f in product_files], ignore_index=True)

# load promotions
promo_path = os.path.join(base_dir, "data", "promotion")
promo_files = [os.path.join(promo_path, f) for f in os.listdir(promo_path) if f.endswith(".parquet")]
promo = pd.concat([pd.read_parquet(f) for f in promo_files], ignore_index=True)

# load sites 
site_path = os.path.join(base_dir, "data", "site")
site_files = [os.path.join(site_path, f) for f in os.listdir(site_path) if f.endswith(".parquet")]
sites = pd.concat([pd.read_parquet(f) for f in site_files], ignore_index=True)

# quebec only
sites = sites[sites['site_state_desc']=='Quebec']


# sample sites 15% of 1426
np.random.seed(666)
sampled_sites = np.random.choice(sites.site_number.unique(), size=215, replace=False)
#%%#

# define range
date_range=[(datetime(2025,1, 13) + timedelta(days=i)).strftime('%Y-%m-%d') for i in range((datetime(2025, 2, 12) - datetime(2025, 1, 13)).days + 1)]

# define matching list
filtered_categories = {
    'Fuel', 'NON-ALIMENTAIRE', 'REVUES ET JOURNAUX', 'DEPENSES',
    'REVENUS DE SERVICE (NON-INV)', 'SERVICE POSTAL', 'CARTES-CADEAUX',
    'BILLETS DE LOTERIE', 'SANTE ET BEAUTE', 'PROPANE EN VRAC',
    'CARTES-CADEAUX COUCHE-TARD', 'CARTES TELEPHONIQUES', 
    'BILLETS DE TRANSPORT', 'DEPENSES SBT', 'RETOUR BOUTEILLE', 
    'CARTES-BANNIERES', 'PRODUITS LEVEE DE FONDS', 'LAVE-AUTO', 
    'CARTES PREP INVENTAIRE', 'Unknown', 'BUREAU DE POSTE', 'TOO GOOD TO GO','CIGARETTES','TABAC'
}
all_categories = set(products['department_desc'].unique())

food_categories = all_categories - filtered_categories
matching_items = products[products['department_desc'].isin(food_categories)]['item_desc'].unique()

# loop through each date in the range to fetch data
data = {}

for date in tqdm(date_range, desc="Processing Dates", unit="date"):
    print(f'====================Process Start: date={date}=========================')
    # load from data
    df_single_date = process_single_date_pandas(
        date_str=date,
        base_dir=base_dir,
        products=products,
        promo=promo,
        sampled_sites=sampled_sites
    )

    # convert to indiv products with record id
    df_single_date['tuples'] = df_single_date.apply(lambda row: list(zip(row['item_desc'], row['quantity_sold'])), axis=1)
    exploded_df = df_single_date.explode('tuples')
    del df_single_date

    # split the tuples into 'item' and 'quantity' columns
    exploded_df[['item', 'quantity']] = pd.DataFrame(exploded_df['tuples'].tolist(), index=exploded_df.index)
    exploded_df = exploded_df.drop(columns=['item_desc', 'quantity_sold', 'tuples'])

    # group by original index and item, then sum quantities
    grouped = exploded_df.groupby([exploded_df.index, 'item'])['quantity'].sum().reset_index()

    ## matching
    # keep only `level_0` values that have at least one matching item
    valid_levels = grouped[grouped["item"].isin(matching_items)]["level_0"].unique()

    # filter the original DataFrame to keep only valid levels
    grouped = grouped[grouped["level_0"].isin(valid_levels)]

    # pivot to binary encoded sales record
    grouped = grouped.set_index(['level_0', 'item'])['quantity']

    # unstack the MultiIndex to create columns from original rows
    sales_encoded = grouped.unstack(level=0, fill_value=0)

    # add to dict
    data[date] = sales_encoded
    print(f'====================Process End: date={date}=========================')

    # delete to save memory
    del exploded_df,grouped,sales_encoded

# save dict
from joblib import dump

# Save the large dictionary to a file
dump(data, os.path.join(base_dir, "data_processed", "sales_record_dict.pkl"))

print("✅ Dictionary saved successfully!")

# save
pd.Series(sampled_sites).to_csv(os.path.join(base_dir, "data_processed", "sampled_sites.csv"), index=False)
# concat
# print('====================Concat Start=========================')
# combined_df = pd.concat(
#     [df.reindex(columns=pd.MultiIndex.from_product([df.columns, [date]], names=['transaction', 'date'])) 
#      for date, df in data.items()],
#     axis=1
# ).fillna(0)



