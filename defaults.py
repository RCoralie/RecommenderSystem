"""
Default parameters
"""

from os.path import dirname, abspath, join

class Config:

    LOG_PATH = 'sentimentAnalysis.log'

    # Path to the resources file
    RES_PATH = join(dirname(abspath(__file__)),"resources") 
    # Path to the persistent resources file
    RES_PERSIST_PATH = join(RES_PATH,"persist")

    RES_TMP_PATH = join(RES_PATH, "tmp")
    RES_MODEL_PATH = join(RES_PATH,"model") 
    RES_GLOBAL_PATH = join(RES_MODEL_PATH,"global") 
    RES_PREDICTIONS_PATH = join(RES_GLOBAL_PATH,"predictions") 
    RES_EVAL_PATH = join(RES_PERSIST_PATH, "evaluations")

    GLOBAL_MODEL_FILE = join(RES_GLOBAL_PATH, "model_weighs.h5")

    GENRES_FILE = join(RES_PATH, "genres.npy")

    COUNTRIES_FILE = join(RES_PATH, "countries.json")

    # Corpus of Global Vectors. Global Vectors is a model for distributed word representation
    GLOVE_CORPUS_FILE = join(RES_PERSIST_PATH, "glove.6B.50d.txt")
    # Dictionary of Global Vectors, based on previous corpus
    GLOVE_DICT_FILE = join(RES_PATH, "glove_dict.npy")

    OVERVIEWS_TR_FILE = join(RES_PERSIST_PATH, "train_overviews_tr.txt")
    OVERVIEW_MODEL = join(RES_PATH, "overviewModel.d2v")

    # Corpus of opinion words
    OPINION_FILE = join(RES_PERSIST_PATH, "opinionWords.txt")
    # Dictionnary of opinion words based on previous corpus
    OPINION_DICT_FILE = join(RES_PATH, "opinionWordsDict.npy")

    # Positive and negative tweets for the training
    TRAIN_TWITTER_NEG_TR_FILE = join(RES_PERSIST_PATH, "train_twitter_neg_tr.txt")
    TRAIN_TWITTER_POS_TR_FILE = join(RES_PERSIST_PATH, "train_twitter_pos_tr.txt")

    # Positive and negative tweets for the testing
    TEST_TWITTER_NEG_TR_FILE = join(RES_PERSIST_PATH, "test_twitter_neg_tr.txt")
    TEST_TWITTER_POS_TR_FILE = join(RES_PERSIST_PATH, "test_twitter_pos_tr.txt")

    # Sentiment analysis model trained
    SENTIMENT_ANALYSIS_MODEL = join(RES_PATH, "sentimentAnalysisModel.h5")

    LABEL_TRAIN_ABTRACTS = 'TRAIN_ABSTRACTS'

    LABEL_TEST_NEG = 'TEST_NEG'

    LABEL_TEST_POS = 'TEST_POS'

    LABEL_TRAIN_NEG = 'TRAIN_NEG'

    LABEL_TRAIN_POS = 'TRAIN_POS'

    TEST = join(RES_EVAL_PATH, "moviesEvaluated-simple.json")

    NB_PREDICTIONS = 4

