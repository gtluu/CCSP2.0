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
from IPython.utils import io


def calculate_descriptors(input_list, input_type='InChI'):
    if input_type == 'InChI':
        mol_list = [Chem.MolFromInchi(i) for i in input_list]
    elif input_type == 'SMILES':
        mol_list = [Chem.MolFromSmiles(i) for i in input_list]
    elif input_type == 'CID':
        compounds_list = pcp.get_compounds(input_list, 'cid')
        mol_list = [Chem.MolFromSmiles(i.isomeric_smiles) for i in compounds_list]
    calc = Calculator(descriptors, ignore_3D=True)
    with io.capture_output() as captured:
        descriptor_list = calc.pandas(mol_list)
    return descriptor_list


def variable_assigner(train_book,
                      test_book,
                      target_book,
                      column_title='Input',
                      train_input_type='InChI',
                      test_input_type='InChI',
                      target_input_type='InChI'):
    x_train = calculate_descriptors(train_book[column_title], train_input_type)
    x_test = calculate_descriptors(test_book[column_title], test_input_type)
    x_target = calculate_descriptors(target_book[column_title], test_input_type)
    y_train = list(train_book['CCS'])
    y_test = list(train_book['CCS'])
    return x_train, y_train, x_test, y_test, x_target


def clean_up_descriptors(input_frame):
    return input_frame.copy().select_dtypes(['number'])


def drop_constant_column(input_frame):
    return input_frame.loc[:, (input_frame != input_frame.iloc[0]).any()]


def remove_outlier(input_frame1, input_frame2, threshold):
    input_frame1_tmp = input_frame1.copy()
    input_frame2_tmp = input_frame2.copy()
    removed_count = 0
    for i in input_frame2.columns:
        val = statistics.mean(input_frame2[i]) - threshold * statistics.stdev(input_frame2[i])
        if max(input_frame1[i]) > val or min(input_frame1[i]) < val:
            input_frame1_tmp.drop(columns=i)
            input_frame2_tmp.drop(columns=i)
            removed_count += 1
    output_frame1 = input_frame1_tmp
    output_frame2 = input_frame2_tmp
    return output_frame1, output_frame2


def svr_model_linear(x_train, y_train, c_list=[2**(i) for i in range(-8, -1)], epsilon_list=[0.5, 1, 5]):
    gsc = GridSearchCV(estimator=SVR(kernel='linear'),
                       param_grid={'C': c_list, 'epsilon': epsilon_list},
                       cv=5,
                       scoring='neg_root_mean_squared_error',
                       verbose=False,
                       n_jobs=-1)
    grid_result = gsc.fit(x_train, y_train)
    best_params = grid_result.best_params_
    best_svr = SVR(kernel='linear',
                   C=best_params['C'],
                   epsilon=best_params['epsilon'],
                   coef0=0.1,
                   shrinking=True,
                   tol=0.001,
                   cache_size=500,
                   verbose=False,
                   max_iter=-1)
    scoring = {'abs_error': 'neg_mean_absolute_error',
               'squared_error': 'neg_mean_squared_error'}
    scores = cross_validate(best_svr,
                            x_train,
                            y_train,
                            cv=5,
                            scoring=scoring,
                            return_train_score=True,
                            n_jobs=-1)
    y_train_cross_validation = cross_val_predict(best_svr,
                                                 x_train,
                                                 y_train,
                                                 cv=5,
                                                 n_jobs=-1,
                                                 verbose=False)
    return best_svr, scores, grid_result, y_train_cross_validation
