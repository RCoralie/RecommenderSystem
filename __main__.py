import sys
import argparse
import logging
import tensorflow as tf

from os.path import isfile, join
from .defaults import Config
from .utils import GloveDict, CreateCorpusOfAbtracts, D2VOnCorpus, apiTMDB
from .model import Classifier, Predictor, ManageModel, AccuracyComparator

tf.logging.set_verbosity(tf.logging.ERROR)

params = {"titles" : False,
          "rating" : True,
          "overviews" : True,
          "keywords" : False,
          "genres" : True,
          "actors" : False,
          "directors" : True,
          "compagnies" : True,
          "language" : False,
          "belongs" : True,
          "runtime" : True,
          "date" : False }


def process_args(args, defaults):

    parser = argparse.ArgumentParser()
    parser.prog = 'sentimentAnalysis'
    subparsers = parser.add_subparsers(help='Subcommands.')

    # Global arguments
    parser.add_argument('--log-path', dest='log_path',
                        type=str, default=defaults.LOG_PATH,
                        help=('Log file path, default=%s' % (defaults.LOG_PATH)))

    # Init resources
    parser_init = subparsers.add_parser('init', help='Create all resources needed if they don\'t already exist.')
    parser_init.set_defaults(phase='init')

    # Training
    parser_train = subparsers.add_parser('train', help='Training model on user movies corpus.')
    parser_train.set_defaults(phase='train')
    parser_train.add_argument('--inputData', dest='data', type=str, help=('Filename of the input data in resources/model repertory, to train model.'))
    parser_train.add_argument('--modelName', dest='modelName', type=str, help=('Name of the trained model in resources/model repertory (for both .json & .h5)'))

    # Prediction
    parser_predict = subparsers.add_parser('predict', help='Predicts n movies thanks to a pre-trained model.')
    parser_predict.set_defaults(phase='predict')
    parser_predict.add_argument('--nb', dest='nb', type=int, default=Config.NB_PREDICTIONS, help=('Number of films to recommend., default = %s' % Config.NB_PREDICTIONS))
    parser_predict.add_argument('--modelName', dest='modelName', type=str, help=('Name of the pre-trained model in resources/model repertory (for both .json & .h5)'))

    # Models comparator
    parser_compare = subparsers.add_parser('compare', help='Compare the accuracy of different models.')
    parser_compare.set_defaults(phase='compare')
    parser_compare.add_argument('--perceptron', dest='doPerceptron', type=bool, default=True, help=('Include the perceptron in the evaluation, default = True'))
    parser_compare.add_argument('--svm', dest='doSVM', type=bool, default=True, help=('Include the SVM in the evaluation, default = True'))
    parser_compare.add_argument('--classifier', dest='doClassifier', type=bool, default=True, help=('Include the classifier in the evaluation, default = True'))
    parser_compare.add_argument('--filenames', dest='filenames', nargs='+', type=str, help=('Name of the files in resources/persist/evaluations, to run the evaluation'))

    parameters = parser.parse_args(args)
    return parameters


def main(args=None):

    if args is None:
        args = sys.argv[1:]

    parameters = process_args(args, Config)
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s',
        filename=parameters.log_path)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    if parameters.phase == 'init':
        # If the file is not present in the resource, creates it 
        if not isfile(Config.GLOVE_DICT_FILE):
            print 'Creation of the glove dictionnary file...'
            gloveDict.createGloveDic()
        if not isfile(Config.OVERVIEWS_TR_FILE):
            print 'Create the corpus of overviews :'
            CreateCorpusOfAbtracts.createCorpus(Config.OVERVIEWS_TR_FILE)
        if not isfile(Config.OVERVIEW_MODEL):
            print 'Create the D2V model on the overwiews corpus :'
            D2VOnCorpus.createD2VModel()

    if parameters.phase == 'train':
        print 'Training model on user movies corpus :'
        model = Classifier.preprocessDataTrainModel(join(Config.RES_EVAL_PATH, parameters.data + '.json'), **params)
        ManageModel.saveModel(parameters.modelName, model)

    if parameters.phase == 'predict':
        print 'Predicts %d movies thanks to a pre-trained model :' %(parameters.nb)
        model = ManageModel.loadModel(parameters.model)
        predictions = Predictor.suggestNMovies(modelName, parameters.nb, **params)
        for movie in predictions:
            print apiTMDB._format(movie["title"])

    if parameters.phase == 'compare':
        print 'Compare the accuracy of different models :'
        scoreK, scoreP , scoreSVM = AccuracyComparator.testClassifier(parameters.filenames, doKeras=parameters.doClassifier, doSVM=parameters.doSVM, doPerceptron=parameters.doPerceptron)
        print 'The classifier keras has an average accuracy of ', scoreK
        print 'The classifier perceptron has an average accuracy of ', scoreP
        print 'The classifier SVM has an average accuracy of ', scoreSVM


if __name__ == "__main__":
    main()
