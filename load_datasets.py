import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from itertools import product
import math
import csv

def open_dataset_1headerrow(filepath, filename):
    fullpath = Path(filepath) / filename

    df_raw = pd.read_csv(fullpath, 
                     encoding = "latin1", 
                     sep = ";", 
                     skiprows = 2,
                     header = None)
    
    row0 = list(df_raw.iloc[0])

    for x in row0[:]:
        if not x.strip():
            row0.remove(x)
    
    empty = [" ", " ", " "]

    new_header = empty + row0

    df = df_raw.iloc[2:]

    df.reset_index(drop = True, inplace = True)

    df.columns = new_header

    column1_raw = list(df.iloc[:,1])
    column2_raw = list(df.iloc[:,2])

    df = df.iloc[2:, 3:]

    column1 = [x for x in column1_raw if not (isinstance(x, float) and math.isnan(x))]
    for x in column1[:]:
        if not x.strip():
            column1.remove(x)
    column2 = [x for x in column2_raw if not (isinstance(x, float) and math.isnan(x))]

    num_ages = int(len(column2)/3)

    df1 = df.iloc[:num_ages]
    df2 = df.iloc[num_ages+1:2*num_ages+1]
    df3 = df.iloc[2*num_ages+2:3*num_ages+2]

    df = pd.concat([df1, df2, df3])

    df.reset_index(drop = True, inplace = True)

    column1_new = []

    for x in column1:
        i = 1
        while i < 9:
            column1_new.append(x)
            i += 1

    df.insert(loc = 0,
        column = 'Sex',
        value = column1_new)
    df.insert(loc = 1,
              column = 'Age',
              value = column2)

    df.set_index(['Sex', 'Age'], inplace = True)

    return df

def open_dataset_2headerrows(filepath, filename):
    fullpath = Path(filepath) / filename
    df_raw = pd.read_csv(fullpath, 
                     encoding = "latin1", 
                     sep = ";", 
                     skiprows = 3,
                     header = None)
    
    row0 = list(df_raw.iloc[0])

    row1_raw = list(df_raw.iloc[1])

    for x in row0[:]:
        if not x.strip():
            row0.remove(x)
    
    row1 = row1_raw[3:6]

    new_header = [x1 + " - " + x2 for x1, x2 in product(row0, row1)]

    empty = [" ", " ", " "]

    new_header = empty + new_header

    df = df_raw[2:]. copy()

    df.reset_index(drop = True, inplace = True)

    df.columns = new_header

    column1_raw = list(df.iloc[:,1])
    column2_raw = list(df.iloc[:,2])

    df = df.iloc[2:, 3:]

    column1 = [x for x in column1_raw if not (isinstance(x, float) and math.isnan(x))]
    for x in column1[:]:
        if not x.strip():
            column1.remove(x)
    column2 = [x for x in column2_raw if not (isinstance(x, float) and math.isnan(x))]

    num_ages = int(len(column2)/3)

    df1 = df.iloc[:num_ages]
    df2 = df.iloc[num_ages+1:2*num_ages+1]
    df3 = df.iloc[2*num_ages+2:3*num_ages+2]

    df = pd.concat([df1, df2, df3])

    df.reset_index(drop = True, inplace = True)

    column1_new = []

    for x in column1:
        i = 1
        while i < 9:
            column1_new.append(x)
            i += 1

    df.insert(loc = 0,
        column = 'Sex',
        value = column1_new)
    df.insert(loc = 1,
              column = 'Age',
              value = column2)

    df.set_index(['Sex', 'Age'], inplace = True)

    return df

def open_dataset_gender_2headerrow(filepath, filename):
    fullpath = Path(filepath) / filename

    df_rows = pd.read_csv(fullpath, 
                          nrows = 2, 
                          encoding = "latin1",
                          sep = ";",
                          header = None)
    header = df_rows.values.tolist()
    
    new_header = str(header[0][0] + " " + header[1][0])

    df_raw = pd.read_csv(fullpath, 
                     encoding = "latin1", 
                     sep = ";", 
                     skiprows = 3,
                     header = None)

    row0 = list(df_raw.iloc[0])

    for x in row0[:]:
        if not x.strip():
            row0.remove(x)

    column1 = df_raw.iloc[:,1]

    ages = list(column1.dropna())
    
    for x in ages[:]:
        if not x.strip():
            ages.remove(x)

    df_new = df_raw.iloc[2:,2:]

    df_new = df_new.dropna()

    both_gender = df_new.iloc[:,0]
    men = df_new.iloc[:,1]
    women = df_new.iloc[:,2]

    ages = ages + ages + ages

    gender = [x for x in row0 for _ in range(8)]
    
    df_concat = pd.concat([both_gender, men, women], ignore_index = True).to_frame(name = new_header)

    df_concat.insert(loc = 0,
        column = 'Sex',
        value = gender)
    df_concat.insert(loc = 1,
              column = 'Age',
              value = ages)

    df_concat.set_index(['Sex', 'Age'], inplace = True)

    return df_concat

def open_dataset_gender_1headerrow(filepath, filename):
    fullpath = Path(filepath) / filename

    df_rows = pd.read_csv(fullpath, 
                          nrows = 1, 
                          encoding = "latin1",
                          sep = ";",
                          header = None)
    header = df_rows.values.tolist()
    
    new_header = str(header[0][0])

    df_raw = pd.read_csv(fullpath, 
                     encoding = "latin1", 
                     sep = ";", 
                     skiprows = 2,
                     header = None)

    row0 = list(df_raw.iloc[0])

    for x in row0[:]:
        if not x.strip():
            row0.remove(x)

    column1 = df_raw.iloc[:,1]

    ages = list(column1.dropna())
    
    for x in ages[:]:
        if not x.strip():
            ages.remove(x)

    df_new = df_raw.iloc[2:,2:]

    df_new = df_new.dropna()

    both_gender = df_new.iloc[:,0]
    men = df_new.iloc[:,1]
    women = df_new.iloc[:,2]

    ages = ages + ages + ages

    gender = [x for x in row0 for _ in range(8)]
    
    df_concat = pd.concat([both_gender, men, women], ignore_index = True).to_frame(name = new_header)

    df_concat.insert(loc = 0,
        column = 'Sex',
        value = gender)
    df_concat.insert(loc = 1,
              column = 'Age',
              value = ages)

    df_concat.set_index(['Sex', 'Age'], inplace = True)

    return df_concat

filepath = Path().resolve()

filename = "KV2PC2.csv"
df_kv2pc2 = open_dataset_1headerrow(filepath, filename)
df_kv2pc2.to_csv('KV2PC2_clean.csv', sep = ";")

filename = "KV2SC1.csv"
df_kv2sc1 = open_dataset_1headerrow(filepath, filename)
df_kv2sc1.to_csv('KV2SC1_clean.csv', sep = ";")

filename = "KV2SC2.csv"
df_kv2sc2 = open_dataset_1headerrow(filepath, filename)
df_kv2sc2.to_csv('KV2SC2_clean.csv', sep = ";")

filename = "KV2SM1.csv"
df_kv2sm1 = open_dataset_1headerrow(filepath, filename)
df_kv2sm1.to_csv('KV2SM1_clean.csv', sep = ";")

filename = "KV2SP2.csv"
df_kv2sp2 = open_dataset_1headerrow(filepath, filename)
df_kv2sp2.to_csv('KV2SP2_clean.csv', sep = ";")

filename = "KV2SP5.csv"
df_kv2sp5 = open_dataset_1headerrow(filepath, filename)
df_kv2sp5.to_csv('KV2SP5_clean.csv', sep = ";")

filename = "KV2MUS1.csv"
df_kv2mus1 = open_dataset_1headerrow(filepath, filename)
df_kv2mus1.to_csv('KV2MUS1_clean.csv', sep = ";")

filename = "KV2NYH1.csv"
df_kv2nyh1 = open_dataset_1headerrow(filepath, filename)
df_kv2nyh1.to_csv('KV2NYH1_clean.csv', sep = ";")

filename = "KV2AI2.csv"
df_kv2ai2 = open_dataset_1headerrow(filepath, filename)
df_kv2ai2.to_csv('KV2AI2_clean.csv', sep = ";")

filename = "KV2BIB1.csv"
df_kv2bib1 = open_dataset_1headerrow(filepath, filename)
df_kv2bib1.to_csv('KV2BIB1_clean.csv', sep = ";")

filename = "KV2BIB2.csv"
df_kv2bib2 = open_dataset_1headerrow(filepath, filename)
df_kv2bib2.to_csv('KV2BIB2_clean.csv', sep = ";")

filename = "KV2BK2.csv"
df_kv2bk2 = open_dataset_1headerrow(filepath, filename)
df_kv2bk2.to_csv('KV2BK2_clean.csv', sep = ";")

filename = "KV2BRN1.csv"
df_kv2brn1 = open_dataset_2headerrows(filepath, filename)
df_kv2brn1.to_csv('KV2BRN1_clean.csv', sep = ";")

filename = "KV2FOR1.csv"
df_kv2for1 = open_dataset_1headerrow(filepath, filename)
df_kv2for1.to_csv('KV2FOR1_clean.csv', sep = ";")

filename = "KV2FR2.csv"
df_kv2fr2 = open_dataset_1headerrow(filepath, filename)
df_kv2fr2.to_csv('KV2FR2_clean.csv', sep = ";")

filename = "KV2FR3.csv"
df_kv2fr3 = open_dataset_1headerrow(filepath, filename)
df_kv2fr3.to_csv('KV2FR3_clean.csv', sep = ";")

filename = "KV2FR4.csv"
df_kv2fr4 = open_dataset_1headerrow(filepath, filename)
df_kv2fr4.to_csv('KV2FR4_clean.csv', sep = ";")

filename = "KV2FS3.csv"
df_kv2fs3 = open_dataset_1headerrow(filepath, filename)
df_kv2fs3.to_csv('KV2FS3_clean.csv', sep = ";")

filename = "KV2LIT2.csv"
df_kv2lit2 = open_dataset_1headerrow(filepath, filename)
df_kv2lit2.to_csv('KV2LIT2_clean.csv', sep = ";")

filename = "KV2LIT3.csv"
df_kv2lit3 = open_dataset_1headerrow(filepath, filename)
df_kv2lit3.to_csv('KV2LIT3_clean.csv', sep = ";")

filename = "KV2MK2.csv"
df_kv2mk2 = open_dataset_1headerrow(filepath, filename)
df_kv2mk2.to_csv('KV2MK2_clean.csv', sep = ";")

filename = "KV2MK3.csv"
df_kv2mk3 = open_dataset_1headerrow(filepath, filename)
df_kv2mk3.to_csv('KV2MK3_clean.csv', sep = ";")

filename = "KV2MKS2.csv"
df_kv2mks2 = open_dataset_1headerrow(filepath, filename)
df_kv2mks2.to_csv('KV2MKS2_clean.csv', sep = ";")

filename = "KV2MO1.csv"
df_kv2mo1 = open_dataset_1headerrow(filepath, filename)
df_kv2mo1.to_csv('KV2MO1_clean.csv', sep = ";")

filename = "KV2MO3.csv"
df_kv2mo3 = open_dataset_1headerrow(filepath, filename)
df_kv2mo3.to_csv('KV2MO3_clean.csv', sep = ";")

filename = "KV2NYH2.csv"
df_kv2nyh2 = open_dataset_1headerrow(filepath, filename)
df_kv2nyh2.to_csv('KV2NYH2_clean.csv', sep = ";")

filename = "KV2NYH3.csv"
df_kv2nyh3 = open_dataset_1headerrow(filepath, filename)
df_kv2nyh3.to_csv('KV2NYH3_clean.csv', sep = ";")

filename = "KV2SP3.csv"
df_kv2sp3 = open_dataset_gender_2headerrow(filepath, filename)
df_kv2sp3.to_csv('KV2SP3_clean.csv', sep = ";")

filename = "KV2SP8.csv"
df_kv2sp8 = open_dataset_gender_1headerrow(filepath, filename)
df_kv2sp8.to_csv('KV2SP8_clean.csv', sep = ";")

filename = "KV2AHOV.csv"
df_kv2ahov = open_dataset_1headerrow(filepath, filename)
df_kv2ahov.to_csv('KV2AHOV_clean.csv', sep = ";")