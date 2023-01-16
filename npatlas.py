import pandas as pd
import datetime
import pickle
from ccsp2.arguments import *
from ccsp2.data_io import *
from ccsp2.model import *
from ccsp2.predict import *


def get_args():
    parser = argparse.ArgumentParser()

    optional = parser.add_argument_group('Optional Parameters')
    optional.add_argument('--output', help='prediction results directory', default='', type=str)

    predict = parser.add_argument_group('Prediction Parameters')
    predict.add_argument('--model', help='model to be used to predict CCS. if not specified, a default model is used',
                         default='models/UnifiedCCSCompendium_cleaned_2022-10-25/model.ccsp2', type=str)

    return vars(parser.parse_args())


def get_npatlas_df():
    npatlas = pd.read_csv('https://www.npatlas.org/static/downloads/NPAtlas_download.tsv', sep='\t')
    colnames = ['npaid',
                'compound_names',
                'compound_m_plus_h',
                'compound_m_plus_na',
                'compound_inchi',
                'compound_smiles']
    npatlas = npatlas[colnames]
    npatlas = npatlas.rename(columns={'npaid': 'ID',
                                      'compound_names': 'Compound',
                                      'compound_m_plus_h': 'M+H',
                                      'compound_m_plus_na': 'M+Na',
                                      'compound_inchi': 'InChI',
                                      'compound_smiles': 'SMILES'})

    npatlas_protonated = npatlas[['ID', 'Compound', 'M+H', 'InChI', 'SMILES']]
    npatlas_protonated = npatlas_protonated.rename(columns={'ID': 'ID',
                                                            'Compound': 'Compound',
                                                            'M+H': 'mz',
                                                            'InChI': 'InChI',
                                                            'SMILES': 'SMILES'})
    npatlas_protonated['Adduct'] = ['M+H'] * len(npatlas_protonated.index)

    npatlas_sodiated = npatlas[['ID', 'Compound', 'M+Na', 'InChI', 'SMILES']]
    npatlas_sodiated = npatlas_sodiated.rename(columns={'ID': 'ID',
                                                        'Compound': 'Compound',
                                                        'M+Na': 'mz',
                                                        'InChI': 'InChI',
                                                        'SMILES': 'SMILES'})
    npatlas_sodiated['Adduct'] = ['M+Na'] * len(npatlas_sodiated.index)

    npatlas = pd.concat([npatlas_protonated, npatlas_sodiated])
    npatlas['Charge'] = ['1'] * len(npatlas.index)

    return npatlas


def predict_npatlas(target_book):
    # set args
    args = get_args()
    args['workflow'] = 'predict'
    args['identifier'] = 'SMILES'
    args['train'] = ''
    args['test'] = ''
    args['split_percentage'] = 50
    args['model_fname'] = 'model'
    args['plot'] = True

    target_input_type, target_input_errors = check_inputs(target_book, column_title=args['identifier'])
    if len(target_input_errors) > 0:
        # instead of exiting, remove error entries from each book
        target_book = target_book[~target_book[args['identifier']].isin(target_input_errors)]
    x_target = variable_assigner(target_book, column_title=args['identifier'], input_type=target_input_type,
                                 book_is_target=True)

    with open(args['model'], 'rb') as model_file:
        models = pickle.load(model_file)
    initial_model = models[0]
    rfe_model = models[1]

    initial_prediction = initial_ccs_prediction(initial_model, x_target, outlier_removal=False, threshold=1000)
    rfe_prediction = rfe_ccs_prediction(rfe_model, initial_prediction['x_target_clean'],
                                        initial_prediction['x_train_clean'], rfe_model['rfecv'])

    target_book_output = target_book.copy()
    #target_book_output['Target CCS Prediction'] = initial_prediction['y_target_predicted']
    target_book_output['Target CCS Prediction RFE VS'] = rfe_prediction['y_target_predicted_rfe']

    target_book_output.to_csv(os.path.join(args['output'], 'target_book_output.csv'), index=False)


if __name__ == '__main__':
    npatlas = get_npatlas_df()
    predict_npatlas(npatlas)
