import pandas as pd
import logging
import numpy as np
df = pd.read_csv('/mnt/Fire_dataset_cleaned.csv',index_col=0)
with open('/mnt/Fire_dataset_cleaned.csv', 'r') as f:
    print(f.read(500))
logging.info(f"DF Columns: {list(df.columns)}")
print(f"df columns: {df.columns}")
df.columns
df.drop(['day','month','year'], axis=1, inplace=True)
df['Classes']= np.where(df['Classes']== 'not fire',0,1)
df.to_csv('/mnt/out/Fire_dataset_cleaned.csv', index=False)
print("New CSV file saved!")
