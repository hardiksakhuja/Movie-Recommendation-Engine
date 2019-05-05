from ast import literal_eval
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
df1 = pd.read_csv('tmdb_5000_credits.csv')
df2 = pd.read_csv('tmdb_5000_movies.csv')

df1.columns = ['id','Title','cast','crew']
df2 = pd.merge(df1,df2,on='id')

print(type(df1['cast']))

features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    df2[feature] = df2[feature].apply(literal_eval)

def get_director(x):
    for i in x:
        if i['job']=='Director':
            return i['name']
    return np.nan

def get_list(x):
    if isinstance(x,list):
        names = [i['name'] for i in x]
        if len(names)>3 :
            names=names[:3]
        return names
    return []

df2['director'] = df2['crew'].apply(get_director)

features = ['cast','keywords','genres']
for feature in features:
    df2[feature] = df2[feature].apply(get_list)

#def clean_data(x):
#    print(type(x))
#    if isinstance(x, list):
#        #print(type(x))
#        for i in x:
#            print(type(x))
#            print(type(i))

#def clean_data(x):
#    list12 = []
#    if isinstance(x, list):
#        #return [str.lower(i.replace(" ", "")) for i in x]
#        for i in x:
#            i = str(i.lower())
#            if i == " ":
#                i=""
#                list12.append(i)
#            else:
#                list12.append(i)
#        return list12
#
#    else:
#        if isinstance(x, str):
#            return str.lower(x.replace(" ", ""))
#        else:
#            return ''

    #    for i in x:
            #return i.str().lower().replace(" ","")
        #return [str(i.lower().replace(" ", "")) for i in x]
#    else:
       #Check if director exists. If not, return empty string
#        if isinstance(x, str):
#            return str.lower(x.replace(" ", ""))
#        else:
#            return ''

def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''


features = ['cast','keywords','genres','director']
for feature in features:
    df2[feature] = df2[feature].apply(clean_data)

def create_soup(x):
    return ' '.join(x['keywords'])+' '+' '.join(x['cast'])+' '+x['director']+' '+' '.join(x['genres'])

df2['soup'] = df2.apply(create_soup , axis=1)

# Import CountVectorizer and create the count matrix


count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df2['soup'])
# Compute the Cosine Similarity matrix based on the count_matrix


cosine = cosine_similarity(count_matrix, count_matrix)
# Reset index of our main DataFrame and construct reverse mapping as before
df2 = df2.reset_index()
indices = pd.Series(df2.index, index=df2['title'])

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
    movie_indices = [i[0] for i in sim_scores]
    return df2['title'].iloc[movie_indices]
    #print(diff)

print(get_recommendations(input('enter name')) )
