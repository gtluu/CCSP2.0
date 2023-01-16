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


def run_model_workflow(args):
    if args['test'] != '':
        train_book, test_book = import_training_data(args['train'], test_book_path=args['test'])
    elif args['test'] == '':
        train_book, test_book = import_training_data(args['train'], split_percentage=args['split_percentage'])
    train_input_type, train_input_errors = check_inputs(train_book, column_title=args['identifier'])
    test_input_type, test_input_errors = check_inputs(test_book, column_title=args['identifier'])
    if len(train_input_errors) + len(test_input_errors) > 0:
        # instead of exiting, remove error entries from each book
        train_book = train_book[~train_book[args['identifier']].isin(train_input_errors)]
        test_book = test_book[~test_book[args['identifier']].isin(test_input_errors)]
    x_train, y_train = variable_assigner(train_book, column_title=args['identifier'], input_type=train_input_type)
    x_test, y_test = variable_assigner(test_book, column_title=args['identifier'], input_type=test_input_type)

    initial_model = train_initial_model(x_train, y_train, x_test, y_test, outlier_removal=False, threshold=1000)
    rfecv = rfe_variable_selection(initial_model['x_train_scaled'],
                                   initial_model['y_train'],
                                   initial_model['grid_results'],
                                   plot=args['plot'])
    rfe_model = train_rfe_model(initial_model['x_train_clean'],
                                initial_model['y_train'],
                                initial_model['x_test_clean'],
                                initial_model['y_test'],
                                rfecv)

    with open(os.path.join(args['output'], args['model_fname'] + '.ccsp2'), 'wb') as model_file:
        pickle.dump((initial_model, rfe_model), model_file)

    train_book_output, test_book_output = train_book.copy(), test_book.copy()
    train_book_output['Calibration CCS Prediction'] = initial_model['y_train_predicted']
    train_book_output['Cross-Validation CCS Prediction'] = initial_model['y_train_cross_validation']
    test_book_output['Validation CCS Prediction'] = initial_model['y_test_predicted']
    train_book_output['Calibration CCS Prediction RFE VS'] = rfe_model['y_train_predicted_rfe']
    train_book_output['Cross-Validation CCS Prediction RFE VS'] = rfe_model['y_train_cross_validation_rfe']
    test_book_output['Validation CCS Prediction RFE VS'] = rfe_model['y_test_predicted_rfe']

    train_book_output.to_csv(os.path.join(args['output'], 'train_book_output.csv'), index=False)
    test_book_output.to_csv(os.path.join(args['output'], 'test_book_output.csv'), index=False)

    # warning: plotting is interactive
    if args['plot']:
        summary_plot_all = summary_plot(y_train,
                                        y_test,
                                        initial_model['y_train_predicted'],
                                        initial_model['y_train_cross_validation'],
                                        initial_model['y_test_predicted'],
                                        labelsize=12,
                                        legendsize=12,
                                        titlesize=14,
                                        textsize=12)
        prediction_plot_calibration = prediction_plot(y_train,
                                                      initial_model['y_train_predicted'],
                                                      train_book,
                                                      hover_column=['Compound'],
                                                      title_string="Calibration Prediction")
        prediction_plot_cross_validation = prediction_plot(y_train,
                                                           initial_model['y_train_cross_validation'],
                                                           train_book,
                                                           hover_column=['Compound'],
                                                           title_string="Cross-Validation Prediction")
        prediction_plot_validation = prediction_plot(y_test,
                                                     initial_model['y_test_predicted'],
                                                     test_book,
                                                     hover_column=['Compound'],
                                                     title_string="Validation Prediction")
        model_diagnostic_plot = model_diagnostics_plot(initial_model['x_test_clean'],
                                                       initial_model['model'])
        summary_plot_rfe = summary_plot(y_train,
                                        y_test,
                                        rfe_model['y_train_predicted_rfe'],
                                        rfe_model['y_train_cross_validation_rfe'],
                                        rfe_model['y_test_predicted_rfe'],
                                        labelsize=12,
                                        legendsize=12,
                                        titlesize=14,
                                        textsize=12)
        prediction_plot_calibration_rfe = prediction_plot(y_train,
                                                          rfe_model['y_train_predicted_rfe'],
                                                          train_book,
                                                          hover_column=['Compound'],
                                                          title_string="Calibration Prediction")
        prediction_plot_cross_validation_rfe = prediction_plot(y_train,
                                                               rfe_model['y_train_cross_validation_rfe'],
                                                               train_book,
                                                               hover_column=['Compound'],
                                                               title_string="Cross-Validation Prediction")
        prediction_plot_validation_rfe = prediction_plot(y_test,
                                                         rfe_model['y_test_predicted_rfe'],
                                                         test_book,
                                                         hover_column=['Compound'],
                                                         title_string="Validation Prediction")
        model_diagnostic_plot_rfe = model_diagnostics_plot(rfe_model['x_test_rfe'],
                                                           rfe_model['model_rfe'])
        plot_list = [summary_plot_all,
                     summary_plot_rfe]
        plot_names = ["summary_plot_all",
                      "summary_plot_rfe"]
        for i in range(len(plot_list)):
            save_location = os.path.join(args['output'], plot_names[i] + '.svg')
            plot_list[i].savefig(save_location)
        plot_list = [prediction_plot_calibration,
                     prediction_plot_cross_validation,
                     prediction_plot_validation,
                     model_diagnostic_plot,
                     prediction_plot_calibration_rfe,
                     prediction_plot_cross_validation_rfe,
                     prediction_plot_validation_rfe,
                     model_diagnostic_plot_rfe]
        plot_names = ["prediction_plot_calibration",
                      "prediction_plot_cross_validation",
                      "prediction_plot_validation",
                      "model_diagnostic_plot",
                      "prediction_plot_calibration_rfe",
                      "prediction_plot_cross_validation_rfe",
                      "prediction_plot_validation_rfe",
                      "model_diagnostic_plot_rfe"]
        for i in range(len(plot_list)):
            save_location = os.path.join(args['output'], plot_names[i] + '.svg')
            plot_list[i].write_image(save_location)