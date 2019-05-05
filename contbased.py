import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

df1 = pd.read_csv('tmdb_5000_credits.csv')
df2 = pd.read_csv('tmdb_5000_movies.csv')

df1.columns = ['id','Title','cast','crew']
df2 = pd.merge(df1,df2,on='id')

print(df2['overview'].head(5))

tfidf = TfidfVectorizer(stop_words='english')
df2['overview'] = df2['overview'].fillna('')
tfidf_matrix = tfidf.fit_transform(df2['overview'])
print(tfidf_matrix.shape)

cosine = linear_kernel(tfidf_matrix,tfidf_matrix)

indices = pd.Series(df2.index, index=df2['title']).drop_duplicates()

def get_recommendations(title, cosine=cosine):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores,key=lambda x: x[1],reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]
    #diff = sim_scores[:12]
    print(sim_scores)
    #print(diff)

    # Get the movie indices
    #movie_indices=[]
    movie_indices = [i[0] for i in sim_scores]
    #for i in sim_scores:
        #movie_indices.append(i[0])

    # Return the top 10 most similar movies
    return df2['title'].iloc[movie_indices]

movie = input("enter movie name \n")
print(get_recommendations(movie))
