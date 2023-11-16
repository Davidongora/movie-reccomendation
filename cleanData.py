import pandas as pd
import numpy as np

data = """User,Movie,Rating
Alice,Star Wars,5
Frank,The Godfather,4
...
Zane,Star Wars,4"""

from io import StringIO
df = pd.read_csv(StringIO(data))

df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce') 
df['Rating'] = df['Rating'].mask(df['Rating'] > 5, np.nan)  

df['Rating'].fillna(df['Rating'].mean(), inplace=True)
print(df)
