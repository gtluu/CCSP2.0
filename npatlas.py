import pandas as pd
import datetime


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

#npatlas.to_csv('data/npatlas_' + str(datetime.datetime.today().date()) + '.csv', index=False)
