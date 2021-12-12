#!/usr/bin/env python

'''
In this first experiment, you will use the pre-trained Word2Vec model called word2vec-google-news-300 to
compute the closest synonym for each word in the dataset. First, use gensim.downloader.load to load the
word2vec-google-news-300 pretrained embedding model. Then use the similarity method from Gensim to
compute the cosine similarity between 2 embeddings (2 vectors) and find the closest synonym to the questionword.
'''

# standard libs
import time
import logging
import numpy as np

# external libs
import gensim.downloader as api

# globals
DETAILS = 'details'
ANALYSIS = 'analysis'
SIMILARITY = 'similarity'
SYNONYMS_FILE = 'data/synonyms.csv'
ANALYSIS_FILE = 'output/' + ANALYSIS + '.csv'
TEST_DATA_SIZE = 80
RUN_SIMILARITY = False


# logging setup
def setup_logger(name, path, level=logging.INFO, header=''):
    log = logging.getLogger(name)
    formatter = logging.Formatter('%(message)s')
    file = logging.FileHandler(path, mode='w')
    file.setFormatter(formatter)
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    log.setLevel(level)
    log.addHandler(file)
    log.addHandler(stream)
    log.info(header)


def logger(msg, name=DETAILS):
    log = logging.getLogger(name)
    log.info(msg)


def run_model(model_name):
    # ensure log files reset per run
    header = 'question,answer,guess,label'
    if RUN_SIMILARITY:
        header += 'max_similarity,min_similarity,avg_similarity,std_similarity'
    setup_logger(f"{DETAILS}-{model_name}", f"output/{model_name}-{DETAILS}.csv", header=header)

    print('Loading: ' + model_name)

    start = time.time()
    model = api.load(model_name)
    word_vectors = model.index_to_key  # used later for checking if certain words included or not
    model_dimension = model.vector_size
    model_corpus_arr = model_name.split('-')[1:-1]
    model_corpus = ' '.join(model_corpus_arr)
    model_load_time = time.time() - start
    model_similarity_stats = []

    print('\nTook ' + str(model_load_time) + ' seconds to load ' + model_name + ' model')

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

    with open(SYNONYMS_FILE) as f:
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
        line_similarity = []

        for guess in options:
            try:
                print('comparing "' + question_word + '" with "' + guess + '"')
                similarity = model.similarity(question_word, guess)
                line_similarity.append(similarity)
                print('similarity:', similarity)
                print('current max_similarity:', max_similarity)

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

        if len(line_similarity) == 0:
            line_similarity.append(-1)

        model_similarity_stats.append(line_similarity)
        min_similarity = min(line_similarity)
        avg_similarity = np.average(line_similarity)
        std_similarity = np.std(line_similarity)

        print('best guess:', best_guess)
        print('label:', label)
        print(f"avg similarity: {avg_similarity} (std: {std_similarity})")
        print(f"max similarity: {max_similarity} / min: {min_similarity}")

        log_line += f"{best_guess},{label}"
        if RUN_SIMILARITY:
            log_line += f",{max_similarity},{min_similarity},{avg_similarity},{std_similarity}"
        logger(log_line, f"{DETAILS}-{model_name}")

    # stats needed for analysis file
    vocab_size = len(word_vectors)
    num_questions_not_guessed = TEST_DATA_SIZE - guess_ctr
    model_accuracy = (correct_ctr / num_questions_not_guessed)

    print('guess_ctr', guess_ctr)
    print('correct_ctr', correct_ctr)
    print('num_questions_not_guessed', num_questions_not_guessed)

    if RUN_SIMILARITY:
        #model_total_similarity = [x for x in sum(model_similarity_stats, []) if x >= 0]
        model_guess_similarity = [x for x in [max(sub) for sub in model_similarity_stats] if x >= 0]
        min_guess_similarity = min(model_guess_similarity)
        max_guess_similarity = max(model_guess_similarity)
        avg_guess_similarity = np.average(model_guess_similarity)
        std_guess_similarity = np.std(model_guess_similarity)
        print(f"avg guess similarity: {avg_guess_similarity} (std: {std_guess_similarity})")
        print(f"max guess similarity: {max_guess_similarity} / min: {min_guess_similarity}")

    log_line = model_corpus + '-' + str(model_dimension) + ',' + model_name + ',' + str(vocab_size) + ','
    log_line += str(correct_ctr) + ',' + str(num_questions_not_guessed) + ',' + str(model_accuracy)
    if RUN_SIMILARITY:
        log_line += f",{max_guess_similarity},{min_guess_similarity},{avg_guess_similarity},{std_guess_similarity}"

    logger(log_line, ANALYSIS)
    if RUN_SIMILARITY:
        logger(f"{model_name}," + ','.join(str(x) for x in model_guess_similarity), SIMILARITY)


def main():
    # setup shared analysis csv
    header = 'corpus-emsize,filename,vocabulary,correct,questions,accuracy'
    if RUN_SIMILARITY:
        header += ',max_guess_similarity,min_guess_similarity,avg_guess_similarity,std_guess_similarity'
    setup_logger(ANALYSIS, ANALYSIS_FILE, header=header)

    if RUN_SIMILARITY:
        setup_logger(SIMILARITY, f"output/{SIMILARITY}.csv")

    run_model('word2vec-google-news-300')  # OG (variant 0)

    # different corpus, same embedding sizes
    run_model('glove-wiki-gigaword-200')  # variant 1
    run_model('glove-twitter-200')

    # same corpus, different embedding sizes
    run_model('glove-twitter-25')
    run_model('glove-twitter-100')


if __name__ == '__main__':
    main()
