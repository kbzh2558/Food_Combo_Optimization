import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns

from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

from scipy.stats.mstats import winsorize

def winsorize_series(s, limits=(0.01, 0.01)):
    return pd.Series(winsorize(s, limits=limits), index=s.index)

def get_product_elasticities(df, product, time, competitors, alphas,winsorize):

    df = df.copy()
    
    if (winsorize[0]):
        df['log_price'] = winsorize_series(df['log_price'], limits=(winsorize[1], winsorize[1]))
        df['residuals'] = winsorize_series(df['residuals'], limits=(winsorize[1], winsorize[1]))
    
    # Drop rows with missing target or predictors
    X_cols = [col for col in df.columns if col.startswith('log_price')]
    df = df.dropna(subset=['residuals'] + X_cols)

    if df.empty or df.shape[0] < len(X_cols) + 2:
        return {
            'product': product,
            'ols_elasticity': np.nan,
            'ols_pvalue': np.nan,
            'ols_significant': False,
            'ridge_elasticity': np.nan,
            'ridge_alpha': np.nan,
            'ridge_elasticity_debiased': np.nan,
            'ridge_alpha_debiased': np.nan,
        }

    # Run models
    ols_result = estimate_elasticity(df, method='OLS')
    #ridge_result = estimate_elasticity(df, method='Ridge', alphas=alphas)
    #ridge_result_debias = estimate_elasticity(df, method='Ridge_Debias', alphas=alphas)

    return {
        'product': product,
        'ols_elasticity': ols_result['elasticity'],
        'ols_pvalue': ols_result['p-value'],
        'ols_significant': ols_result['significant'],
        #'ridge_elasticity': ridge_result['elasticity'],
        #'ridge_alpha': ridge_result['alpha'],
        #'ridge_elasticity_debiased': ridge_result_debias['elasticity'],
        #'ridge_alpha_debiased': ridge_result_debias['alpha'],
    }

def estimate_elasticity(df, method='OLS', alphas=None):
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    df = df.copy()

    if df.empty or df.dropna().shape[0] < 3:
        return {
            'elasticity': np.nan,
            'p-value': np.nan,
            'significant': False,
            'alpha': None
        }

    # Extract predictors
    X_cols = [col for col in df.columns if col.startswith('log_price')]
    X = df[X_cols].dropna()
    if X.empty:
        return {
            'elasticity': np.nan,
            'p-value': np.nan,
            'significant': False,
            'alpha': None
        }

    y = df.loc[X.index, 'residuals']

    if method.upper() == 'OLS':
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        return {
            'elasticity': model.params.get('log_price'),
            'p-value': model.pvalues.get('log_price'),
            'significant': model.pvalues.get('log_price') <= 0.05,
            'alpha': None
        }

    
    elif method.upper() == 'RIDGE':
        if alphas is None:
            raise ValueError("You must provide alphas for RidgeCV.")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = RidgeCV(alphas=alphas, cv=cv)
        model.fit(X_scaled, y)

        coef = model.coef_[X.columns.get_loc('log_price')]
        std_dev = scaler.scale_[X.columns.get_loc('log_price')]
        elasticity = coef / std_dev

        return {
            'elasticity': elasticity,
            'p-value': None,
            'significant': None,
            'alpha': model.alpha_
        }
    
    elif method.upper() == 'RIDGE_DEBIAS':

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        ridge = RidgeCV(alphas=alphas, cv=cv)
        ridge.fit(X_scaled, y)

        # Ridge coefficient vector (standardized space)
        beta_ridge = ridge.coef_

        # Closed-form bias correction:
        lambda_opt = ridge.alpha_
        XtX = X_scaled.T @ X_scaled
        p = X.shape[1]
        I = np.identity(p)

        # Apply debiasing formula
        correction_term = lambda_opt * np.linalg.inv(XtX + lambda_opt * I) @ beta_ridge
        beta_debiased = beta_ridge + correction_term

        # Rescale back to original units
        std_devs = scaler.scale_
        idx = X.columns.get_loc('log_price')
        elasticity = beta_debiased[idx] / std_devs[idx]

        return {
            'elasticity': elasticity,
            'p-value': None,
            'significant': None,
            'alpha': lambda_opt
        }

    else:
        raise ValueError("Method must be 'OLS' or 'Ridge'")

def get_product_weights(df):
    
    df = df.copy()
    
    product_qty = sales.groupby(['category','label','brand','product_desc']).agg({'quantity_sold':'sum'}).reset_index()
    label_qty = sales.groupby(['category','label']).agg(category_label_qty=('quantity_sold','sum')).reset_index()

    product_weights = product_qty.merge(label_qty,on=['category','label'],how='left')
    product_weights['weight'] = product_weights['quantity_sold'] / product_weights['category_label_qty']
    
    return product_weights

def add_competitor_effects(df, agg_sales, competitors, level='brand', time='W'):
    df = df.copy()
    
    if len(competitors) == 0:
        return df

    id_col = 'brand' if level == 'brand' else 'product_desc'
    time_keys = ['year', 'month'] if time == 'M' else ['wk_year', 'wk_month', 'week_of_year']

    for i, comp in enumerate(competitors):
        comp_df = (
            agg_sales[agg_sales[id_col] == comp][[*time_keys, 'log_price']]
            .rename(columns={'log_price': f'log_price_competitor_{i+1}'})
        )
        df = df.merge(comp_df, on=time_keys, how='left')

    return df

def get_competitors(corr_matrix, max_cross=3, min_corr=0.3, max_inter_corr=0.7):

    output = {}

    # remove weak correlations
    filtered = corr_matrix.where(corr_matrix >= min_corr)

    for item in filtered.index:
        selected = []
        sorted_corrs = filtered.loc[item].drop(item).dropna().sort_values(ascending=False)

        for comp in sorted_corrs.index:
            
            # check if comp is too correlated with already selected brands/products
            if any(filtered.loc[comp, sel] > max_inter_corr for sel in selected):
                continue
            selected.append(comp)
            if len(selected) >= max_cross:
                break

        output[item] = selected

    return output

def get_corr_matrix(df, category, level='product', time='W'):
    # Filter dataframe for the given category
    df_cat = df[df['category'] == category].copy()

    # Choose identifier based on level
    id_col = 'brand' if level == 'brand' else 'product_desc'

    if time == 'W':
        group_keys = ['category', 'wk_year', 'wk_month', 'week_of_year', id_col]
        index_keys = ['category', 'wk_year', 'wk_month', 'week_of_year']
    else:
        group_keys = ['category', 'year', 'month', id_col]
        index_keys = ['category', 'year', 'month']

    # Aggregate log_price by summing 
    agg_df = df_cat.groupby(group_keys)['log_price'].sum().reset_index()

    # Pivot the aggregated dataframe: rows are time periods, columns are products/brands
    pivot_df = agg_df.pivot_table(index=index_keys,
                                  columns=id_col,
                                  values='log_price')

    # Compute correlation matrix among products based on the pivoted log_price data
    corr_matrix = pivot_df.corr()

    return corr_matrix

def add_residuals(df, time):
    
    df = df.copy()

    if time == 'W':
        
        # Validate required columns
        required_cols = ['week_of_year', 'wk_year', 'wk_month']
        if not all(col in df.columns for col in required_cols):
            raise ValueError("Missing one or more required columns: 'week_of_year', 'wk_year', 'wk_month'")

        # Weekly seasonality
        df['sin_week'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
        df['cos_week'] = np.cos(2 * np.pi * df['week_of_year'] / 52)

        # Monthly seasonality
        df['sin_month'] = np.sin(2 * np.pi * df['wk_month'] / 12)
        df['cos_month'] = np.cos(2 * np.pi * df['wk_month'] / 12)

        X = df[['wk_year', 'sin_week', 'cos_week', 'sin_month', 'cos_month']].dropna()
        X = sm.add_constant(X)
        y = df.loc[X.index, 'log_qty']

        model = sm.OLS(y, X).fit()
        df['residuals'] = model.resid

    elif time == 'M':
        
        # Validate required columns
        required_cols = ['month', 'year']
        if not all(col in df.columns for col in required_cols):
            raise ValueError("Missing one or more required columns: 'month', 'year'")

        # Monthly seasonality
        df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
        df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)

        X = df[['year', 'sin_month', 'cos_month']].dropna()
        X = sm.add_constant(X)
        y = df.loc[X.index, 'log_qty']

        model = sm.OLS(y, X).fit()
        df['residuals'] = model.resid

    else:
        raise ValueError("time must be 'W' or 'M'")

    return df

def agg_sales(df, agg):
    
    df = df.copy()
    
    df['revenue'] = (df['price'] * df['quantity_sold'])
    
    df = (df.groupby(agg)
         .agg(qty_sold=('quantity_sold','sum'),revenue=('revenue','sum'))
         .reset_index())
    
    df["weighted_avg_price"] = df['revenue'] / df['qty_sold']
    
    df['log_qty'] = np.log1p(df['qty_sold']) 
    df['log_price'] = np.log1p(df['weighted_avg_price'])
    
    df.drop(columns=['revenue','qty_sold','weighted_avg_price'], inplace=True)
    
    return df

def process_sales(sales_file_copy, years, pres_threshold, cov_threshold, time_metric, verbose):
    
    
    if (time_metric == 'W'):
        
        df = sales_file_copy[(sales_file_copy['wk_year'].isin(years))].copy()

        df['week_index'] = ((df['week_start'] - df['week_start'].min()).dt.days // 7).astype(int)        
        inWeek = (df.groupby(['item_id', 'week_index']).size()
                                                       .gt(pres_threshold)  
                                                       .unstack(fill_value=False)
                                                       .astype(int)).copy()
        
        inWeek.reset_index(inplace=True)
        inWeek['coverage'] = round(inWeek.iloc[:, 1:].sum(axis=1)/df['week_index'].max(),2)
        
        mask = list(inWeek[(inWeek['coverage'] < cov_threshold)]
                    ['item_id'].unique())
                
    else:
        
        df = sales_file_copy[(sales_file_copy['year'].isin(years))].copy()

        df['month_index'] = (df['year'] - df['year'].min()) * 12 + df['month']        
        inMonth = (df.groupby(['item_id', 'month_index']).size()
                                                         .gt(pres_threshold)  
                                                         .unstack(fill_value=False)
                                                         .astype(int)).copy()

        inMonth.reset_index(inplace=True)
        inMonth['coverage'] = round(inMonth.iloc[:, 1:].sum(axis=1)/df['month_index'].max(),2)

        mask = list(inMonth[(inMonth['coverage'] < cov_threshold)]
                    ['item_id'].unique())
    
    filtered = df[~(df['item_id'].isin(mask))]
    
    if (verbose): 
        
        print(f"{len(mask)} products masked")
        print(f"{len(filtered['item_id'].unique())} products remaining")
        
        print(pd.DataFrame(filtered.groupby(['category','label']).agg({'item_id':'nunique'})))

    return filtered

# loading sales file and preprocessing
sales_file = pd.read_csv("qc_sales_for_elasticity.csv")

sales_file['business_date'] = pd.to_datetime(sales_file['business_date'])
sales_file.drop(columns=["Unnamed: 0"], inplace = True)
sales_file['category'] = sales_file['category_desc'].str.replace(r'-C|-|/| |\*', '_', regex=True)
sales_file['category'] = sales_file['category'].str.strip('_')
sales_file['category'] = sales_file['category'].replace('BREUVAGES_FROIDS', 'BOISSONS_GAZEUSES')

sales_file['year'] = sales_file['business_date'].dt.year

sales_file['week'] = sales_file['business_date'].dt.strftime('%W').astype(int)
sales_file['week_start'] = sales_file['business_date'] - pd.to_timedelta(sales_file['business_date'].dt.dayofweek, unit='D')
sales_file['week_of_year'] = sales_file['week_start'].dt.strftime('%W').astype(int)

sales_file['wk_year'] = sales_file['week_start'].dt.year
sales_file['wk_month'] = sales_file['week_start'].dt.month

labels = sales_file[['category','brand','label']].drop_duplicates().reset_index()

product_monthly = ['category', 'brand', 'product_desc','year', 'month']
product_weekly = ['category', 'brand', 'product_desc','wk_year', 'wk_month','week_of_year']

sales = process_sales(sales_file.copy(),[2022,2023,2024], 4, 0.8, 'W', False)
aggregated_sales = agg_sales(sales, product_weekly)
    
product_weights = get_product_weights(sales).drop(columns=['brand','quantity_sold','category_label_qty'])

results = []
alphas = np.logspace(-3, 2, 20)

product_categories = (
    aggregated_sales.drop_duplicates('product_desc')[['product_desc', 'category']]
    .set_index('product_desc')['category']
    .to_dict()
)

for product in aggregated_sales['product_desc'].unique():

    category = product_categories[product]
    category_sales = aggregated_sales[aggregated_sales['category'] == category]

    corr_matrix = get_corr_matrix(category_sales, category, level='product', time='W')
    competitor_dict = get_competitors(corr_matrix, max_cross=3, min_corr=0.2, max_inter_corr=0.9)

    competitors = competitor_dict.get(product, [])
    
    product_agg_sales = category_sales[category_sales['product_desc'] == product].copy()
    product_agg_sales = add_residuals(product_agg_sales, 'W')
    product_agg_sales = add_competitor_effects(product_agg_sales, category_sales, competitors, 'product', 'W')

    result = get_product_elasticities(product_agg_sales, product, 'W', competitors, alphas, (True, 0.05))
    results.append(result)
    
results = pd.DataFrame(results)
results = results.rename(columns={'product':'product_desc'})
results = product_weights.merge(results,on='product_desc',how='left')

results['weighted_OLS'] = results['weight'] * results['ols_elasticity']

elasticities = results
results = results.groupby(['category','label']).agg({'weighted_OLS':'sum', 'product_desc':'nunique'}).reset_index()

results.to_excel("final_elasticity_results.xlsx")