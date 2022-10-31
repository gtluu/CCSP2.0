import ctypes
import math
import statistics
import time
import tkinter as tk
from multiprocessing import freeze_support
import os
from sys import stdout
from tkinter.filedialog import asksaveasfile
from tkinter import *
from tkinter import messagebox, simpledialog, filedialog
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pubchempy as pcp
import sklearn
from mordred import Calculator, descriptors
from rdkit import Chem
from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, cross_val_predict, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.utils import shuffle
import io


def import_training_data(train_book_path, split_percentage=50, test_book_path=None):
    if test_book_path is None:
        train_test_book = pd.read_csv(train_book_path)
        rand_master = train_test_book.reindex(np.random.permutation(train_test_book.index))
        train_book = rand_master.sample(frac=split_percentage/100)
        test_book = pd.concat([train_test_book, train_book]).drop_duplicates(keep=False)
    elif test_book_path is not None:
        train_book = pd.read_csv(train_book_path)
        test_book = pd.read_csv(test_book_path)
    return train_book, test_book


def import_query_data(target_book_path):
    return pd.read_csv(target_book_path)


def check_inputs(book, column_title='Input'):
    input_errors = []
    if isinstance(list(book[column_title])[0], int):
        input_type = 'CID'
        for i in book[column_title]:
            compound = pcp.get_compounds(i, 'cid')
            if Chem.MolFromSmiles(compound[0].isomeric_smiles) is None:
                input_errors.append(i)
    elif 'InChI' in list(book[column_title])[0]:
        input_type = 'InChI'
        for i in book[column_title]:
            if Chem.MolFromInchi(i) is None:
                input_errors.append(i)
    else:
        input_type = 'SMILES'
        for i in book[column_title]:
            if Chem.MolFromSmiles(i) is None:
                input_errors.append(i)
    return input_type, input_errors


def get_compounds_from_cid(input_list):
    cid_list = [int(i) for i in input_list]
    reconstructed_cids = []
    reconstructed_compounds = []
    comp_val = 0
    initial_val = 0
    while comp_val < len(cid_list):
        initial_val = comp_val
        comp_val += 100
        reconstructed_cids.extend(cid_list[initial_val:comp_val])
        compounds_vals = pcp.get_compounds(cid_list[initial_val:comp_val], 'cid')
        reconstructed_compounds.extend(compounds_vals)
        progress = 100 * (initial_val) / len(cid_list)
        if comp_val > len(cid_list):
            progress = 100
    return reconstructed_compounds
