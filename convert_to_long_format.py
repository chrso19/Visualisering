import pandas as pd
from pathlib import Path
import numpy as np

def load_descriptions(file_name, sep = ';'):
    df = pd.read_csv(file_name, sep=sep)
    df = df.dropna(axis=1,how ='all')
    df['Multiple answers?'] = df['Multiple answers?'].map({'Yes': True, 'No': False, None: np.nan})
    df['Multiple answers?'] = df['Multiple answers?'].astype('boolean')
    df = df.rename(columns={'Question/prompt asked':'question', 'Dataset name':'source_file',
                            'Short description':'description', 'Multiple answers?':'multiple answers'})
    return df

def load_all_csvs(folder_path, sep = ';'):
    """
    Loads all csvs in a folder and saves them in a list.
    Adds filename as column in each DataFrame.
    
    Args:
        folder_path (str or Path): path to folder
    
    Returns:
        list: list of DataFrames.
    """
    folder = Path(folder_path)
    dataframes = []

    for file_path in folder.glob("*.csv"):  # finds all filenames
        df = pd.read_csv(file_path, sep=sep)
        df['source_file'] = file_path.stem.removesuffix('_clean')  # filename without .csv and '_clean'
        df = df.rename(columns={'Sex':'sex','Age':'age'})
        dataframes.append(df)
        

    return dataframes

def remove_sex_age_total(df):
    df = df[df['sex'] != "Sex, total"]
    df = df[df['age'] != "Age, total"]
    return df.reset_index(drop=True)


def convert_to_long(df):
    # number of data points (percentages) in original df
    original_values = df.drop(columns=['sex','age','source_file']).size

    df_long = pd.melt(df, id_vars=['sex', 'age','source_file'], value_name= 'percentage', var_name= 'answer')
    df_long['percentage'] = pd.to_numeric(df_long['percentage'], errors='coerce')
    
    # number of rows in long table
    melted_values = df_long.shape[0]

    if original_values != melted_values:
        print("Number of rows in long table is wrong")
 
    return df_long


folder = "Datasets"
dfs = load_all_csvs(folder)

print(f"{len(dfs)} files are loaded.")


dfs_long=[]
for df in dfs:
    dfs_long.append(convert_to_long(remove_sex_age_total(df)))
    
combined_df = pd.concat(dfs_long, ignore_index=True) 

desc_df = load_descriptions('dataset_description.txt')

full_df = pd.merge(combined_df, desc_df, how = 'left')
full_df = full_df[['source_file','description','multiple answers','question','answer','sex','age','percentage']]

print(full_df.dtypes)

# check whether adding descriptions messed something up, 
# if not, save

save_csv = True
save_pickle = True
if combined_df.shape[0] == full_df.shape[0]: 
    if save_csv:
        full_df.to_csv("full_data_long_format.csv",sep = ';')
        print('.csv saved!')
    if save_pickle:
        full_df.to_pickle('full_data_long_format.pkl')
        print('.pkl saved!')
