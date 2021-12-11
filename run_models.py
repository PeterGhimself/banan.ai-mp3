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
import logging

# external libs
import gensim.downloader as api


# logging setup

def setup_logger(logger_name, log_file, level=logging.INFO):
    log_setup = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(message)s')
    fileHandler = logging.FileHandler(log_file, mode='a')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    log_setup.setLevel(level)
    log_setup.addHandler(fileHandler)
    log_setup.addHandler(streamHandler)


def logger(msg, logfile):
    if logfile == 0:
        log = logging.getLogger('details')
    elif logfile == 1:
        log = logging.getLogger('analysis')
    else:
        print('Invalid logfile option')
        return

    log.info(msg)


def run_model(model_name):
    SYNONYMS_CSV_FILE = 'data/synonyms.csv'
    DETAILS_FILE = 'output/' + model_name + '-details.csv'
    ANALYSIS_FILE = 'output/' + model_name + '-analysis.csv'
    TEST_DATA_SIZE = 80

    setup_logger('details', DETAILS_FILE, logging.INFO)
    setup_logger('analysis', ANALYSIS_FILE, logging.INFO)

    # ensure log files reset per run
    with open(DETAILS_FILE, 'w') as fout:
        fout.writelines('')
    with open(ANALYSIS_FILE, 'w') as fout:
        fout.writelines('')

    print('Loading: ' + model_name)

    start = time.time()
    model = api.load(model_name)
    word_vectors = model.index_to_key  # used later for checking if certain words included or not
    model_load_time = time.time() - start

    print('\nTook ' + str(model_load_time) + ' seconds to load ' + model_name + 'embedding model')

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

    with open(SYNONYMS_CSV_FILE) as f:
        lines = f.readlines()

    lines = lines[1:]  # strip away first header line
    log_line = ''
    correct_ctr = guess_ctr = 0

    for line in lines:
        print('----------------------')
        line = line.strip()
        tings = line.split(',')
        question_word = tings[0]
        question_answer = tings[1]
        options = tings[2:]

        print('question word:', question_word)
        print('question answer:', question_answer)
        print('guess options:', options)

        log_line = question_word + ',' + question_answer + ','
        best_guess = label = ''
        max_similarity = 0

        for guess in options:
            try:
                print('comparing "' + question_word + '" with "' + guess + '"')
                similarity = model.similarity(question_word, guess)
                print('similarity:', similarity)
                print('max_similarity:', max_similarity)

                if similarity > max_similarity:
                    max_similarity = similarity
                    best_guess = guess

            except KeyError:
                if question_word not in word_vectors:
                    print('question word "' + question_word + '" not found in model')
                    label = 'guess'
                else:
                    at_least_one_found = False
                    for guess in options:
                        if guess not in word_vectors:
                            print('guess "' + guess + '" not found in model')
                        else:
                            at_least_one_found = True

                    if not at_least_one_found:
                        label = 'guess'
            finally:
                if not label == 'guess':
                    if best_guess == question_answer:
                        label = 'correct'
                    else:
                        label = 'wrong'

        if label == 'guess':
            guess_ctr += 1
        elif label == 'correct':
            correct_ctr += 1

        print('best guess:', best_guess)
        print('label:', label)

        log_line += best_guess + ',' + label
        logger(log_line, 0)

    # stats needed for analysis file
    vocab_size = len(word_vectors)
    num_questions_not_guessed = TEST_DATA_SIZE - guess_ctr
    model_accuracy = (correct_ctr / num_questions_not_guessed) * 100
    model_accuracy = round(model_accuracy, 2)

    print('guess_ctr', guess_ctr)
    print('correct_ctr', correct_ctr)
    print('num_questions_not_guessed)', num_questions_not_guessed)

    log_line = model_name + ',' + str(vocab_size) + ',' + str(correct_ctr) + ','
    log_line += str(num_questions_not_guessed) + ',' + str(model_accuracy) + '%'

    logger(log_line, 1)

run_model('word2vec-google-news-300') # OG (variant 0)

# different corpus, same embedding sizes
run_model('glove-wiki-gigaword-200') # variant 1
run_model('glove-twitter-200')

# same corpus, different embedding sizes
run_model('glove-twitter-25')
run_model('glove-twitter-200')

