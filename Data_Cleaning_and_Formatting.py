# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import re

df_dict = {}
fin_dict = {}

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        name = re.sub('\.xlsx', '', filename)
        reject_names = ['9780062381828', '9780545825030', '9780547076690', '9780142414538']
        if (name in reject_names):
            continue
        df_dict[name] = pd.read_excel(os.path.join(dirname, filename))
        
        df_dict[name] = df_dict[name].drop(df_dict[name].columns[0], axis=1)
        df_dict[name] = df_dict[name].rename(columns={df_dict[name].columns[0]: 'Text', df_dict[name].columns[1]: 'Commentary'})
        
        df_dict[name]['ISBN'] = name
        df_dict[name]['Page'] = np.arange(1, len(df_dict[name]) + 1)
        
        fin_dict[name] = df_dict[name].copy(deep=True)
        
        for count, row in enumerate(fin_dict[name].iterrows()):
            if count == 0 or count == 1 or pd.isna(df_dict[name].iat[count, 0]):
                continue
            
            two_page = "" if pd.isna(df_dict[name].iat[count - 2, 0]) else df_dict[name].iat[count - 2, 0]
            one_page = "" if pd.isna(df_dict[name].iat[count - 1, 0]) else df_dict[name].iat[count - 1, 0]
                        
            fin_dict[name].iat[count, 0] = two_page + " " + one_page + " " + df_dict[name].iat[count, 0]

df = pd.DataFrame()

for isbn in fin_dict:
    df = df.append(fin_dict[isbn])

df = df.dropna()
df = df.replace({r'\s+$': '', r'^\s+': ''}, regex=True).replace(r'\n',  ' ', regex=True)
df = df[df.Commentary != '-']
df = df[df["Commentary"].str.contains("Praise")==False]

from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.2, random_state=42)
test, val = train_test_split(test, test_size=0.5, random_state=42)

train.to_csv('CONTEXT_b1Train', index=False)
test.to_csv('CONTEXT_b1Test', index=False)
val.to_csv('CONTEXT_b1Val', index=False)
