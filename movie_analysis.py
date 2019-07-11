#Decomposing text documents with LDA

import pandas as pd 
import numpy as np
import time
import math
import seaborn as sns
import tarfile
import pyprind
import os
import re
import io
import sys

#with tarfile.open('aclImdb_v1.tar.gz', 'r:gz') as tar:
#	tar.extractall()

from explode_function import tidy_split
from matplotlib import pyplot as plt 
from pandas import Series, DataFrame, pivot_table
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


movies = pd.read_csv('movies.csv')

#print(movies.shape)
#print(movies)

#movies.genres = movies.genres.str.split('|')
#print(movies.genres)
movies = tidy_split(movies, 'genres')
#movies = movies.loc[movies['genres'].isin(['Romance', 'Horror'])]

ratings = pd.read_csv('ratings.csv', nrows=10000) #total rows = 20,000,264

ratings['timestamp'] = ratings['timestamp'].apply(lambda x: time.strftime('%Y', time.localtime(x)))

movies = movies.drop('title', axis=1)

ratings = ratings.merge(movies, left_on='movieId', right_on='movieId', how='inner')
#print(ratings.head())
#print(ratings.shape)


ratings = ratings.loc[ratings['genres'].isin(['Sci-Fi', 'Animation', 'Comedy', 'Romance', 'Thriller', 'Horror', 'Musical'])]
#ratings = ratings.loc[ratings['timestamp'].isin(['1995', '1996', '1997', '1998', '1999', '2000'])]
mean_ratings = ratings.groupby(['timestamp', 'genres'])['rating'].aggregate(np.mean)
mean_ratings.rename(columns={'timestamp': 'year'}, inplace=True)
sd_ratings = ratings.groupby(['timestamp', 'genres'])['rating'].aggregate(np.std)
#print(sd_ratings.rename(columns={'timestamp': 'year'}, inplace=True))

ratings2 = ratings.groupby(['movieId', 'timestamp', 'genres'], as_index=False)['rating'].aggregate(np.mean)

#sns.pointplot(x='timestamp', y='rating', hue='genres', data=ratings2, legend=False, ci=None)
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#plt.show()

#ratings2.to_csv('ratings2_output.csv')

ratings3 = ratings.groupby(['userId', 'timestamp', 'genres'], as_index=False)['rating'].aggregate(np.mean)

#sns.pointplot(x='timestamp', y='rating', hue='genres', data=ratings3, legend=False, ci=None)
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#plt.show()

pivot = pivot_table(ratings, values='rating', index=['userId', 'timestamp'], columns='genres')

# Compute the correlation matrix and average over networks
#corr_df = pivot.corr().groupby(level="userId").mean()
#corr_df.index = corr_df.index.astype(int)
#corr_df = corr_df.sort_index().T


# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 6))

#sns.violinplot(x='timestamp', y='rating', hue='genres', data=ratings3, palette='Set3') #, bw=.2, cut=1, linewidth=1)

# Finalize the figure
#ax.set(ylim=(-.7, 1.05))
#sns.despine(left=True, bottom=True)
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

#sns.jointplot(x='Romance', y='Horror', data=pivot, kind="reg")
#sns.jointplot(x='Comedy', y='Horror', data=pivot, kind="hex")
#sns.jointplot(x='Romance', y='Horror', data=pivot, kind="kde", color='r')
#plt.show()

basepath = 'C:\FTG\Python_Scripts\movie_analysis\_aclImdb'

labels = {'pos': 1, 'neg': 0}
pbar = pyprind.ProgBar(5000)
df = pd.DataFrame()
for s in ('test', 'train'):
	for l in ('pos', 'neg'):
		path = os.path.join(basepath, s, l)
		for file in os.listdir(path):
			with io.open(os.path.join(path, file), mode='r', encoding='utf-8') as infile:
				txt = infile.read()
			df = df.append([[txt, labels[l]]], ignore_index=True)
			pbar.update()
df.columns = ['review', 'sentiment']

#np.random.seed(0)
#df = df.reindex(np.random.permutation(df.index))
#df.to_csv('movie_data.csv', index=False, encoding='utf-8')
#print(df.head(3))

df = pd.read_csv('movie_data.csv', encoding='utf-8')

count = CountVectorizer(stop_words='english',
						max_df=.1,
						max_features=500)
X = count.fit_transform(df['review'].values)

lda = LatentDirichletAllocation(n_components=10,
								random_state=123,
								learning_method='batch')
X_topics = lda.fit_transform(X)

n_top_words = 5
feature_names = count.get_feature_names()
for topic_idx, topic in enumerate(lda.components_):
	print("Topic %d:" % (topic_idx +1))
	print(" ".join([feature_names[i]
					for i in topic.argsort()\
						[:-n_top_words - 1:-1]]))

horror = X_topics[:, 5].argsort()[::-1]
for iter_idx, movie_idx in enumerate(horror[:3]):
	print('\nHorror movie #%d:' % (iter_idx + 1))
	print(df['review'][movie_idx][:300], '...')

#df = pd.read_csv('tmdb_5000_movies.csv')
#print(df.head())