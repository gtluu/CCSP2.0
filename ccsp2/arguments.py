import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='training dataset or combined dataset to be split into test and train sets',
                        required=True, type=str)
    parser.add_argument('--test', help='test dataset', default='', type=str)
    parser.add_argument('--query', help='query dataset to predict CCS for', default='', type=str)
    parser.add_argument('--output', help='prediction results', default='', type=str)
    parser.add_argument('--split_percentage', help='percentage to split training dataset into test and train sets',
                        default=50, type=int)
    return vars(parser.parse_args())
