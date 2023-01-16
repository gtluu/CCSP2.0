import pandas as pd
import datetime


def get_coconut_df():
    coconut = pd.read_csv('https://coconut.naturalproducts.net/download/absolutesmiles',
                          sep=' ', header=None, names=['SMILES', 'ID'])
    return coconut

#coconut.to_csv('data/coconut_' + str(datetime.datetime.today().date()) + '.csv', index=False)
