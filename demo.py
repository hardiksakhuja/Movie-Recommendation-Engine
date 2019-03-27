import pandas as pd
import numpy as np

df1 = pd.read_csv('tmdb_5000_credits.csv')
df2 = pd.read_csv('tmdb_5000_movies.csv')

df1.columns = ['id','title','cast','crew']
df2 = df2.merge(df1,on='id')
print(df2.head(5))

C=df2['vote_average'].mean()
print(C)
