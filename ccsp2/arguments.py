import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='training dataset or combined dataset to be split into test and train sets',
                        required=True, type=str)
    parser.add_argument('--test', help='test dataset', default='', type=str)
    parser.add_argument('--query', help='query dataset to predict CCS for', default='', type=str)
    parser.add_argument('--output', help='prediction results directory', default='', type=str)
    parser.add_argument('--split_percentage', help='percentage to split training dataset into test and train sets',
                        default=50, type=int)
    parser.add_argument('--train_test_identifier', help='name of the column containing the compound identifier',
                        choices=['CID', 'InChI', 'SMILES'], default='InChI', type=str)
    parser.add_argument('--target_identifier', help='name of the column containing the compound identifier',
                        choices=['CID', 'InChI', 'SMILES'], default='SMILES', type=str)
    parser.add_argument('--plot', help='output plots during analysis', action='store_true')
    return vars(parser.parse_args())
