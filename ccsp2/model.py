import ctypes
import math
import statistics
import time
import tkinter as tk
from multiprocessing import freeze_support
import os
import pickle
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
from ccsp2.arguments import *
from ccsp2.data_io import *
from ccsp2.model import *
from ccsp2.predict import *


def calculate_descriptors(input_list, input_type='InChI'):
    if input_type == 'InChI':
        mol_list = [Chem.MolFromInchi(i) for i in input_list]
    elif input_type == 'SMILES':
        mol_list = [Chem.MolFromSmiles(i) for i in input_list]
    elif input_type == 'CID':
        compounds_list = pcp.get_compounds(input_list, 'cid')
        #compounds_list = get_compounds_from_cid(input_list)
        mol_list = [Chem.MolFromSmiles(i.isomeric_smiles) for i in compounds_list]
    calc = Calculator(descriptors, ignore_3D=True)
    #with io.capture_output() as captured:
    #    descriptor_list = calc.pandas(mol_list)
    descriptor_list = calc.pandas(mol_list)
    return descriptor_list


def variable_assigner(book, column_title='Input', input_type='InChI', book_is_target=False):
    x = calculate_descriptors(book[column_title], input_type)
    if not book_is_target:
        y = list(book['CCS'])
        return x, y
    elif book_is_target:
        return x


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


def train_initial_model(x_train, y_train, x_test, y_test, outlier_removal=False, threshold=1000):
    x_train_clean = clean_up_descriptors(x_train)
    x_train_clean = drop_constant_column(x_train_clean)
    x_test_clean = clean_up_descriptors(x_test)
    x_train_clean_raw = x_train_clean
    x_test_clean_raw = x_test_clean
    common_columns = [col for col in set(x_train_clean.columns).intersection(x_test_clean.columns)]
    x_train_clean, x_test_clean = x_train_clean[common_columns], x_test_clean[common_columns]
    if outlier_removal:
        x_test_clean, x_train_clean = remove_outlier(x_test_clean, x_train_clean, threshold)
    common_columns = [col for col in set(x_train_clean.columns).intersection(x_test_clean.columns)]
    x_train_clean, x_test_clean = x_train_clean[common_columns], x_test_clean[common_columns]
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_clean)
    x_test_scaled = scaler.transform(x_test_clean)
    model, scores, grid_results, y_train_cross_validation = svr_model_linear(x_train_scaled, y_train)
    model.fit(x_train_scaled, y_train)
    y_train_predicted = model.predict(x_train_scaled)
    y_test_predicted = model.predict(x_test_scaled)
    return {'x_train_clean_raw': x_train_clean_raw,
            'x_test_clean_raw': x_test_clean_raw,
            'x_train_clean': x_train_clean,
            'x_test_clean': x_test_clean,
            'x_train_scaled': x_train_scaled,
            'x_test_scaled': x_test_scaled,
            'model': model,
            'grid_results': grid_results,
            'scaler': scaler,
            'y_train': y_train,
            'y_test': y_test,
            'y_train_predicted': y_train_predicted,
            'y_train_cross_validation': y_train_cross_validation,
            'y_test_predicted': y_test_predicted}


def train_rfe_model(x_train_clean, y_train, x_test_clean, y_test, rfecv):
    x_rfe = x_train_clean[x_train_clean.columns[rfecv.support_]]
    x_test_rfe = x_test_clean[x_train_clean.columns[rfecv.support_]]
    scaler = StandardScaler()
    x_rfe_scaled = scaler.fit_transform(x_rfe)
    x_test_rfe_scaled = scaler.transform(x_test_rfe)
    model_rfe, scores_rfe, grid_result_rfe, y_train_cross_validation_rfe = svr_model_linear(x_rfe_scaled, y_train)
    model_rfe.fit(x_rfe_scaled, y_train)
    y_train_predicted_rfe = model_rfe.predict(x_rfe_scaled)
    y_test_predicted_rfe = model_rfe.predict(x_test_rfe_scaled)
    return {'model_rfe': model_rfe,
            'grid_result_rfe': grid_result_rfe,
            'rfecv': rfecv,
            'scaler': scaler,
            'y_train': y_train,
            'x_test_rfe': x_test_rfe,
            'y_train_predicted_rfe': y_train_predicted_rfe,
            'y_train_cross_validation_rfe': y_train_cross_validation_rfe,
            'y_test_predicted_rfe': y_test_predicted_rfe}
