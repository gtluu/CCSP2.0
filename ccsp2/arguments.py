import argparse
from ccsp2.arguments import *
from ccsp2.data_io import *
from ccsp2.model import *
from ccsp2.predict import *

def get_args():
    parser = argparse.ArgumentParser()

    required = parser.add_argument_group('Required Parameters')
    required.add_argument('--workflow', help='choose workflow to run', required=True, type=str,
                          choices=['all', 'model', 'predict'])

    optional = parser.add_argument_group('Optional Parameters')
    optional.add_argument('--output', help='prediction results directory', default='', type=str)
    optional.add_argument('--identifier', help='name of the column containing the compound identifier',
                          choices=['CID', 'InChI', 'SMILES'], default='InChI', type=str)

    model = parser.add_argument_group('Model Creation Parameters')
    model.add_argument('--train', help='training dataset or combined dataset to be split into test and train sets',
                       default='', type=str)
    model.add_argument('--test', help='test dataset', default='', type=str)
    model.add_argument('--split_percentage', help='percentage to split training dataset into test and train sets',
                       default=50, type=int)
    model.add_argument('--model_fname', help='filename for model to be saved. default is "model"',
                       default='model', type=str)
    model.add_argument('--plot', help='output plots during analysis', action='store_true')

    predict = parser.add_argument_group('Prediction Parameters')
    predict.add_argument('--query', help='query dataset to predict CCS for', default='', type=str)
    predict.add_argument('--model', help='model to be used to predict CCS. if not specified, a default model is used',
                         default='models/UnifiedCCSCompendium_cleaned_2022-10-25/model.ccsp2', type=str)

    return vars(parser.parse_args())
