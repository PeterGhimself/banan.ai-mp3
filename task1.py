#!/usr/bin/env python

'''
In this first experiment, you will use the pre-trained Word2Vec model called word2vec-google-news-300 to
compute the closest synonym for each word in the dataset. First, use gensim.downloader.load to load the
word2vec-google-news-300 pretrained embedding model. Then use the similarity method from Gensim to
compute the cosine similarity between 2 embeddings (2 vectors) and find the closest synonym to the questionword.
'''

# standard libs
import time
import sys

# external libs
import gensim.downloader as api

GOOGLE_NEWS_MODEL = 'word2vec-google-news-300'

print('Loading: ' + GOOGLE_NEWS_MODEL)

start = time.time()
model = api.load(GOOGLE_NEWS_MODEL)
model_load_time = time.time() - start

print('\nTook ' + str(model_load_time) + ' seconds to load GoogleNews embedding model')

print('\nSimilarity of "woman" and "man" --> ' + str(model.similarity('woman', 'man')))

'''
The output of this task should be stored in 2 files:
1. In a file called <model name>-details.csv, for each question in the Synonym Test dataset, in a single line:
(a) the question-word, a comma,
(b) the correct answer-word, a comma
(c) your system’s guess-word, a comma
(d) one of 3 possible labels:
• the label guess, if either question-word or all four guess-words (or all 5 words) were not found in
the embedding model (so if the question-word was present in the model, and at least 1 guess-word
was present also, you should not use this label).
• the label correct, if the question-word and at least 1 guess-word were present in the model, and
the guess-word was correct.
• the label wrong if the question-word and at least 1 guess-word were present in the model, and the
guess-word was not correct.
For example, the file word2vec-google-news-300-details.csv could contain:
enormously,tremendously,uniquely,wrong
provisions,stipulations,stipulations,correct
...
2. In a file called analysis.csv, in a single line:
(a) the model name (clearly indicating the source of the corpus and the vector size), a comma
(b) the size of the vocabulary (the number of unique words in the corpus1
)
(c) the number of correct labels (call this C), a comma
(d) the number of questions that your model answered without guessing (i.e. 80− guess) (call this V ), a
comma
(e) the accuracy of the model (i.e. C
V
)
For example, the file analysis.csv could contain:
word2vec-google-news-300,3000000,44,78,0.5641025641025641

'''
