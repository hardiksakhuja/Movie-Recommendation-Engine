import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df1 = pd.read_csv('tmdb_5000_credits.csv')
df2 = pd.read_csv('tmdb_5000_movies.csv')

df1.columns = ['id','Title','cast','crew']
#df2 = df2.merge(df1,on='id')
df2 = pd.merge(df1,df2,on='id')
print(df2.head(5))

c=df2['vote_average'].mean()
#print(C)
m = df2['vote_count'].quantile(0.9)
print(m)
#qual_movies = df2.copy().loc[df2['vote_count'] >= m]
qual_movies = df2[df2['vote_count'] >= m]
print(qual_movies.shape)

def value(x,m=m,c=c):
    v=x['vote_count']
    r=x['vote_average']

    return (v/(v+m) * r) + (m/(m+v) * c)

qual_movies['score'] = qual_movies.apply(value,axis=1)
q_movies = qual_movies.sort_values('score',ascending=False)
print(q_movies.head(2))

print(q_movies[['Title','vote_count','vote_average','score']].head(10))


pop= df2.sort_values('popularity', ascending=False)
plt.figure(figsize=(12,4))

plt.barh(pop['title'].head(6),pop['popularity'].head(6), align='center',
        color='skyblue')
plt.gca().invert_yaxis()
plt.xlabel("Popularity")
plt.title("Popular Movies")
plt.show()
