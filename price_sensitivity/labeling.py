import pandas as pd
import numpy as np

brands = ['BREUVAGES CHAUDS','BREUVAGES GLACES','CK/CT','CK/CT ARIZONA','CK/CT MONDOUX',
         'COUCHE-TARD MENU','FFF','FONTAINE (POLAR POP)','JOKER MAD']

exclude_departments = ['NON-ALIMENTAIRE',
                        'CARTES-CADEAUX',
                        'BILLETS DE TRANSPORT',
                        'DEPENSES',
                        'REVENUS DE SERVICE (NON-INV)',
                        'CIGARETTES',
                        'TABAC',
                        'CARTES TELEPHONIQUES',
                        'PROPANE EN VRAC',
                        'GNFR DEPENSE',
                        'DEPENSES SBT',
                        'Fuel',
                        'CARTES-BANNIERES',
                        'LAVE-AUTO',
                        'SERVICE POSTAL',
                        'BILLETS DE LOTERIE',
                        'PRODUITS NON SCANNES',
                        'RETOUR BOUTEILLE',
                        'CARTES PREP INVENTAIRE',
                        'EPICERIE',
                        'BIERE',
                        'VIN ET ALCOOL',
                        'Unknown',
                        'SANTE ET BEAUTE',
                        'REVUES ET JOURNAUX',
                        'TOO GOOD TO GO',
                        'PRODUITS LEVEE DU FOND',
                        'CARTES-CADEAUX COUCHE-TARD']

exclude_categories = ['EXTRAS ET ACCOMPAGNEMENTS',
                     'TREMPETTES',
                     'NON-ALIMENTAIRE-C',
                     'PRODUITS LEVEE DE FONDS-C',
                     'RESTAURANT PIZZA PIZZA',
                     'MANDAT POSTE-C',
                     'BUREAU POSTE/FRAIS']

prod1 = pd.read_parquet("product/part-00000-2acd59d2-8f63-4dba-b97a-9bd95235ab57-c000.snappy.parquet")
prod2 = pd.read_parquet("product/part-00000-9944d9a8-d4e3-497a-8a9d-9a8d5c25e5dd-c000.snappy.parquet")
products = pd.concat([prod1, prod2], axis=0, ignore_index=True)
target_categories = products[~(products['department_desc'].isin(exclude_departments))
                    & ~(products['category_desc'].isin(exclude_categories))]['category_desc'].unique()

pb_division = pd.read_excel("PB_DIVISION.xlsx", usecols=['Item','Description','Marque','Catégorie','Sous-Catégorie'])
pb_division.rename(columns={'Item':'item_id',
                           'Marque':'brand',
                           'Catégorie':'category_desc',
                           'Sous-Catégorie':'sub_category_desc',
                           'Description':'product_desc'}, inplace=True)
pb_division = pb_division.drop_duplicates(subset='item_id')

pb_division['label'] = np.where(pb_division['brand'].isin(brands), 'private', 'national')

pb_division.drop(columns=['product_desc'], inplace=True)

qc_sales = pd.read_csv("data_processed/qc_time_series.csv")
qc_sales = qc_sales.merge(pb_division,on='item_id',how='left')
qc_sales.dropna(inplace=True)
qc_sales = qc_sales[qc_sales['category_desc'].isin(target_categories)]

qc_sales.to_csv("qc_sales_for_elasticity.csv")