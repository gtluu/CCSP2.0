import numpy as np
import pandas as pd
import datetime


def get_gnps_df():
    gnps = pd.read_json('https://gnps-external.ucsd.edu/gnpslibraryjson')
    colnames = ['spectrum_id', 'Compound_Name', 'Adduct', 'Precursor_MZ', 'Charge', 'Smiles', 'INCHI']
    gnps = gnps[colnames]
    gnps = gnps.rename(columns={'spectrum_id': 'ID',
                                'Compound_Name': 'Compound',
                                'Adduct': 'Adduct',
                                'Precursor_MZ': 'mz',
                                'Charge': 'Charge',
                                'Smiles': 'SMILES',
                                'INCHI': 'InChI'})

    # prioritizing SMILES formulas here
    gnps['SMILES'].replace('N/A', np.nan, inplace=True)
    gnps['SMILES'].replace(' ', np.nan, inplace=True)
    gnps.dropna(subset=['SMILES'], inplace=True)
    gnps['SMILES'] = gnps['SMILES'].apply(lambda x: x.strip())
    gnps['SMILES'].replace('', np.nan, inplace=True)
    gnps.dropna(subset=['SMILES'], inplace=True)

    return gnps

#gnps.to_csv('data/gnps_' + str(datetime.datetime.today().date()) + '.csv', index=False)
