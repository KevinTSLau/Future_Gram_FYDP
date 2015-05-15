import pdb
import re
import nltk
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.collocations import *
from sklearn.decomposition import NMF
bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()

# change this to read in your data
words = 'For my fitness and post-workout needs I turn to @proteinworld whey protein, not only great quality but a great tasting shake'
feature_vector = {}

split_words = words.split(' ')

#Create a vectorizer
vectorizer = TfidfVectorizer(stop_words = 'english')
#Fit our data into the vectorizer
feature_matrix = vectorizer.fit_transform(split_words)

#Retrieve all the feature names
feature_names = vectorizer.get_feature_names()

num_rows, num_cols = feature_matrix.shape
row_nz, col_nz = feature_matrix.nonzero()

#Create a new feature vector dict mapping feature names to feature value
for row, col in zip(row_nz, col_nz):
	feature_vector[feature_names[col]] = feature_matrix[row, col]

print 'Results from sklearn library using TF-IDF ' + str(feature_vector)  + '\n'

#Create a bigram collocation finder
finder = BigramCollocationFinder.from_words(split_words)

# only bigrams that appear 1+ times (heuristic)
finder.apply_freq_filter(1) 

# return the dynamic amount of n-grams with the highest PMI
important = finder.nbest(bigram_measures.pmi, len(split_words)/3)

print 'Results from NLTK library using bi-gram ' + str(important) + '\n'

#All the features from sklearn is automatically inserted into the final list.
#If the features in bigram happens to be in sklearn, increase their importance.
#Features not in sklearn but in bigram are ignored.
for element_vector in important:
	for element in element_vector:
		if feature_vector.get(element.lower(), False):
			feature_vector[element.lower()] += 1

print 'Final results by combining the two libraries output ' + str(feature_vector) 

