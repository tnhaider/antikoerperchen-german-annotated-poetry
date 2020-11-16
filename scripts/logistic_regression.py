import sys
#import pandas as pd
#import numpy as np
#from sklearn.linear_model import LogisticRegression
#from sklearn.linear_model import LogisticRegressionCV
#from sklearn.linear_model import SGDClassifier

import sys, re
import itertools
import numpy as np
import scipy as sp
import pandas as pd
import random as rn
from random import randint
from sklearn.metrics import f1_score
from statistics import mean
from somajo import SoMaJo

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
#from torchnlp.datasets import imdb_dataset
from sklearn.model_selection import train_test_split
from collections import Counter
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from sklearn.pipeline import FeatureUnion
#from pandas import DataFrameMapper
from sklearn_pandas import DataFrameMapper

#import matplotlib.pyplot as plt
#import torch
#from pytorch_pretrained_bert import BertModel
#from torch import nn
#from torchnlp.datasets import imdb_dataset
#from pytorch_pretrained_bert import BertTokenizer
#from keras.preprocessing.sequence import pad_sequences
#from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
#from torch.optim import Adam
#from torch.nn.utils import clip_grad_norm_
#from IPython.display import clear_output

tokenizer = SoMaJo("de_CMC", split_camel_case=True)

def load_word2vec_model(modelpath):
	model = Word2Vec.load(modelpath)
	return model

def get_w2v_most_similar(model, word, topn=3):
	try:
		vector = model.wv[word]
		#print(word,vector)
		sims = model.wv.most_similar(word, topn=topn)
		#print(word, sims)
		return sims
	except KeyError:
		return None


#### FILEPATH ARGUMENTS
# sys.argv[1]: path to feature table
# sys.argv[2]: path to word2vec model
data = pd.read_csv(sys.argv[1], sep='\t')
word2vec_model = load_word2vec_model(sys.argv[2])
#print(get_w2v_most_similar(word2vec_model, 'Bank'))
#print(data.head())

#### CONFIG
# just comment out what you don't want
# 'baseline' has to be in the list
config = [
	  'baseline', 		# baseline
	  #'measure',		# i_measure
	  #'measure_length',	# penta, tetra, etc.
	  #'period',		# lit. period
	  #'author',		# author name
	  #'kNN'			# thesaurus info from word2vec
	  ]

# set the size of the additional thesaurus synonyms per word
top_k = 2

# Size of test set, thus trainsize = 1-testsize
test_size = 0.7

# Number of crossvalidation folds over the dataset
folds = 10

# Model setup: can be 'fast', 'performance', or 'playground'
model_setup = 'fast'


stanza_texts = {}
stanza_emos = {}
stanza_measures = {}
stanza_periods = {}
stanza_authors = {}
all_instances = zip(data.stanza_id, data.line_text, data.emotion1anno1, data.i_measure, data.period, data.author)
for sid, text, emo, measure, period, author in all_instances:
	#print(sid, text, emo)
	stanza_texts.setdefault(sid, []).append(text)
	stanza_emos.setdefault(sid, []).append(emo)
	stanza_measures.setdefault(sid, []).append(measure)
	stanza_periods.setdefault(sid, []).append(period)
	stanza_authors.setdefault(sid, []).append(author)
X = []
X_f = []
X_knn = []
y = []
for sid, texts in stanza_texts.items():
	text = " ".join(texts)
	X.append(text)

	emos = Counter(stanza_emos[sid]).most_common(1)
	emo = [i[0] for i in emos]
	y.append(emo)

	authors = Counter(stanza_authors[sid]).most_common(1)
	author = authors[0][0]
	author = re.sub(',', '_', author)
	author = re.sub(' ', '_', author)
	periods = Counter(stanza_periods[sid]).most_common(1)
	period = ' '.join([i[0] for i in periods])
	measures = Counter(stanza_measures[sid]).most_common(2)
	measure_top2 = ' '.join(['_'.join(i[0].split('.')) for i in measures])
	measure_length = ' '.join([i[0] for i in measures])

	#X_f.append(measure2 + ' ' + period + ' ' + author)
	
	poem_features = ''

	if 'measure' in config:
		poem_features += ' ' + measure_top2
	if 'measure_length' in config:
		poem_features += ' ' + measure_length
	if 'period' in config:
		poem_features += ' ' + period
	if 'author' in config:
		poem_features += ' ' + author

	X_f.append(poem_features)
	#X_f.append(measure2
	#		+ ' ' + measure3
	#	        + ' ' + period
	#		+ ' ' + author
	#		)

	expanded_text = ''
	if 'kNN' in config:
		tokens = []
		tokenized = tokenizer.tokenize_text([text])
		for sentence in tokenized:
			for token in sentence:
				if token.token_class == 'regular':
					tokens.append(token.text)
		for token in tokens:
			nearest = get_w2v_most_similar(word2vec_model, token, topn=top_k)
			if nearest:
				for i, score in nearest:
					expanded_text += i + ' '
		#print(text)
		#print(expanded_text)
	X_knn.append(expanded_text)	

#print(X)
#print(y)
#train_texts, test_texts, train_labels, test_labels = train_test_split(
#     X, y, test_size=0.2, random_state=142)
#texts = data.line_text
#emotions = data.emotion1anno1
#train_texts, test_texts, train_labels, test_labels = train_test_split(
#     texts, emotions, test_size=0.2, random_state=42)

#train_data_full, test_data_full = imdb_dataset(train=True, test=True)
##print(train_data_full)
#rn.shuffle(train_data_full)
#rn.shuffle(test_data_full)
#train_data = train_data_full[:1000]
#test_data = test_data_full[:1000]


## Get the texts and the labels from data
######################################################
#train_texts, train_labels = list(zip(*map(lambda d: (d['text'], d['sentiment']), train_data)))
#test_texts, test_labels = list(zip(*map(lambda d: (d['text'], d['sentiment']), test_data)))
#print(test_texts)
#print(test_labels)

## Train model and predict
######################################################

#mapper = DataFrameMapper([
#    (['i_measure', 'period', 'author'], None),
#    ('line_text',CountVectorizer(binary=False, ngram_range=(1, 3)))
#])

#X = mapper.fit_transform(data)
#X_column_names = mapper.features[0][0]+mapper.features[1][1].get_feature_names()
#y = data.emotion1anno1

#model = make_pipeline(CountVectorizer(ngram_range=(1,3)), LogisticRegression(solver='lbfgs', multi_class='multinomial'))
c_vec = CountVectorizer(analyzer='word', ngram_range=(1,3))
X2 = c_vec.fit_transform(X)
feature_names_ngrams = c_vec.get_feature_names()
X_columns = feature_names_ngrams
X3 = []
X4 = []

if 'kNN' in config:
	c_vec_expand = CountVectorizer(analyzer='word', ngram_range=(1,1))
	X3 = c_vec_expand.fit_transform(X_knn)
	features_expand = c_vec_expand.get_feature_names()
	X_columns += features_expand

if 'measure' in config or 'measure_length' in config or 'period' in config or 'author' in config:
	c_vec_uni = CountVectorizer(analyzer='word', ngram_range=(1,1))
	X4 = c_vec_uni.fit_transform(X_f)
	unigram_features = c_vec_uni.get_feature_names()
	X_columns += unigram_features

print()
print('##########################################')
print('Tail of Feature Array: ')
print(X_columns[-200:])
print('Number of features:', len(X_columns))

if len(config) == 1 and 'baseline' in config:
	X = X2
if len(config) == 2 and 'kNN' in config and 'baseline' in config:
	X = sp.sparse.hstack((X2, X3), format='csr')
if not 'kNN' in config and len(config) > 1:
	X = sp.sparse.hstack((X2, X4), format='csr')
if len(config) > 2 and 'kNN' in config:
	X = sp.sparse.hstack((X2, X3, X4), format='csr')

#for uni in unigrams:
#	print(uni, get_w2v_most_similar(word2vec_model, uni))
#for x in X2.toarray():
#	print(x)
f1_macros = []
for i in range(folds):
	random_state=randint(0, 200)
	print()
	print('***************************************')
	print('Run No.:', str(i+1), 'of', str(folds))
	print('Config:', config)

	y = np.array(y).ravel() # y.values.ravel()
	train_texts, test_texts, train_labels, test_labels = train_test_split(
	     X, y, test_size=test_size, random_state=random_state)


	if model_setup == 'fast':
		model = LogisticRegression(
					solver='newton-cg', 
					penalty='l2',
					C=1,
					multi_class='multinomial',
					max_iter=1000,
					#multi_class='ovr',
					#n_jobs=31,
					#verbose=1,
					)
	if model_setup == 'performance':
		model = LogisticRegression(
					solver='saga', 
					penalty='elasticnet',
					l1_ratio=0.5, # 0 = l2, 1 = l1
					C=1,
					#multi_class='multinomial',
					multi_class='ovr',
					max_iter=10000,
					n_jobs=31,
					#verbose=1,
					)
	if model_setup == 'playground':
		model = LogisticRegression(
					#solver='newton-cg', 
					#penalty='l2',
					solver='saga', 
					penalty='elasticnet',
					l1_ratio=0.5, # 0 = l2, 1 = l1
					C=1,
					#multi_class='multinomial',
					multi_class='ovr',
					max_iter=10000,
					n_jobs=31,
					#verbose=1,
					)
	print(model)
	model = model.fit(train_texts, train_labels)

	y_pred = model.predict(test_texts)
	f1_macro = f1_score(test_labels, y_pred, average='macro')
	print('F1-macro:', f1_macro)
	print()
	f1_macros.append(f1_macro)
	print(classification_report(test_labels, y_pred))

print()
print('***************************************')
print('Number of runs:', str(i+1))
print('Mean F1-macro over all runs:', mean(f1_macros))

#model = LogisticRegressionCV(cv=5, random_state=0, solver='newton-cg', multi_class='multinomial')
#model = model.fit(X, y)

'''
def convert_label(column):
    # column = data.emotion1anno1
    seen = {}
    data2 = np.array([])
    for i, label in enumerate(column):
        if label in seen:
            data2 = np.append(data2, [seen[label]], axis=0)
        else:
            if len(seen) == 0:
                seen[label] = 1
                data2 = np.append(data2, [1], axis=0)
            else:
                highest = sorted(seen.values())[-1]
                seen[label] = highest + 1
                data2 = np.append(data2, [highest+1], axis=0)
    return data2.reshape(-1, 1)


if __name__ == "__main__":	
	data = pd.read_csv(sys.argv[1], sep='\t')
	data.head()
	#data.dtypes
	#data.shape[0]
	X = convert_label(data.emotion1anno1)
	#y = convert_label(data.s_measure)
	#X = data.emotion1anno1
	y = data.measure
	#print(X)
	#print(y)
	#model = SGDClassifier()
	model = LogisticRegression(solver='newton-cg', multi_class='multinomial')
	model.fit(X,y)
	print(model.score(X,y))
	print(model.classes_)
	print(model.coef_)
	print(model.intercept_)
'''
