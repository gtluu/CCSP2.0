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
from ccsp2.model import *


def initial_ccs_prediction(x_train, y_train, x_test, y_test, x_target, outlier_removal=False, threshold=1000):
    x_train_clean = clean_up_descriptors(x_train)
    x_train_clean = drop_constant_column(x_train_clean)
    x_test_clean = clean_up_descriptors(x_test)
    x_target_clean = clean_up_descriptors(x_target)
    common_columns = [col for col in set(x_train_clean.columns).intersection(x_test_clean.columns).intersection(x_target_clean.columns)]
    x_train_clean, x_test_clean, x_target_clean = x_train_clean[common_columns], x_test_clean[common_columns], x_target_clean[common_columns]
    if outlier_removal:
        x_test_clean, x_train_clean = remove_outlier(x_test_clean, x_train_clean, threshold)
        x_target_clean, x_train_clean = remove_outlier(x_target_clean, x_train_clean, threshold)
    common_columns = [col for col in set(x_train_clean.columns).intersection(x_test_clean.columns).intersection(x_target_clean.columns)]
    x_train_clean, x_test_clean, x_target_clean = x_train_clean[common_columns], x_test_clean[common_columns], x_target_clean[common_columns]
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_clean)
    x_test_scaled = scaler.transform(x_test_clean)
    x_target_scaled = scaler.transform(x_target_clean)
    model, scores, grid_results, y_train_cross_validation = svr_model_linear(x_train_scaled, y_train)
    model.fit(x_train_scaled, y_train)
    y_train_predicted = model.predict(x_train_scaled)
    y_test_predicted = model.predict(x_test_scaled)
    y_target_predicted = model.predict(x_target_scaled)
    return x_train_clean, x_test_clean, x_target_clean, x_train_scaled, model, grid_results, y_train_predicted, y_train_cross_validation, y_test_predicted, y_target_predicted


def regression_metrics(true_values, predicted_values):
    rmse = mean_squared_error(true_values, predicted_values, squared=False)
    mre = statistics.median([abs(true_values[i] - predicted_values[i]) / true_values[i]*100
                             for i in range(len(predicted_values))])
    r2 = r2_score(true_values, predicted_values)
    return rmse, mre, r2


def prediction_plot(true_values, predicted_values, input_book, hover_column=['Compound Name'], title_string=None):
    input_book['Relative Error (%)'] = [round(abs(true_values[i] - predicted_values[i]) / true_values[i]*100, 2)
                                        for i in range(len(true_values))]
    predict_plot_px = px.scatter(input_book,
                                 x=true_values,
                                 y=predicted_values,
                                 color_continuous_scale=px.colors.diverging.RdYlGn_r,
                                 color='Relative Error (%)',
                                 template='none',
                                 hover_data=hover_column,
                                 title=title_string,
                                 trendline='ols')
    predict_plot_px.update_layout(autosize=False,
                                  width=900,
                                  height=500,
                                  hovermode='closest')
    predict_plot_px.update_xaxes(title_test='Measured CCS (<span>&#8491;</span><sup>2</sup>)',
                                 title_font={'size': 15},
                                 title_standoff=25)
    predict_plot_px.update_yaxes(title_text='Predicted CCS (<span>&#8491;</span><sup>2</sup>)',
                                 title_font={'size': 15},
                                 title_standoff=25)
    predict_plot_px.update_layout(hoverlabel_align='auto')
    predict_plot_px.data[1].line.color = 'skyblue'
    predict_plot_px.add_shape(type='line',
                              x0=min(min(true_values), min(predicted_values)),
                              x1=max(max(true_values), max(predicted_values)),
                              y0=min(min(true_values), min(predicted_values)),
                              y1=max(max(true_values), max(predicted_values)),
                              line=dict(dash='dash'))
    predict_plot_px.show()
    return predict_plot_px


def model_diagnostics_plot(x_block, model_name):
    df_name = pd.DataFrame()
    df_name['Descriptor'] = list(x_block.columns)
    model_dx_plot_px = px.scatter(df_name,
                                  x=range(len(list(model_name.coef_[0]))),
                                  y=list(model_name.coef_[0]),
                                  color_discrete_sequence=px.colors.sequential.Turbo,
                                  color=list(model_name.coef_[0]),
                                  template='none',
                                  hover_data=['Descriptor'])
    model_dx_plot_px.update_yaxes(title_text='Feature Weight',
                                  title_font={'size': 15},
                                  title_standoff=25)
    model_dx_plot_px.update_xaxes(title_text='Descriptor Number',
                                  title_font={'size': 15},
                                  title_standoff=25)
    model_dx_plot_px.show()
    return model_dx_plot_px


def rfe_variable_selection(x_train_scaled, y_train, grid_results, plot=False):
    estimator = SVR(C=grid_results.best_params_['C'],
                    cache_size=500,
                    coef0=0.1,
                    epsilon=grid_results.best_params['epsilon'],
                    kernel='linear')
    rfecv = RFECV(estimator,
                  step=5,
                  cv=5,
                  scoring='neg_root_mean_squared_error',
                  min_features_to_select=1,
                  verbose=0,
                  n_jobs=-1)
    rfecv.fit(x_train_scaled, y_train)
    if plot:
        plt.figure()
        plt.xlabel('Number of Features Selected')
        plt.ylabel('Negative root Mean Squared Error ($\AA^2$)')
        x_points = list(range(len(x_train_scaled.columns), 1, -5))
        if 1 not in x_points:
            x_points = x_points + [1]
        plt.plot(x_points[::-1], rfecv.grid_scores_)
        plt.show()
    return rfecv


def rfe_ccs_prediction(x_train_clean, y_train, x_test_clean, y_test, x_target_clean, rfecv):
    x_rfe = x_train_clean[x_train_clean.columns[rfecv.support_]]
    x_test_rfe = x_test_clean[x_train_clean.columns[rfecv.support_]]
    x_target_rfe = x_target_clean[x_train_clean.columns[rfecv.support_]]
    scaler = StandardScaler()
    x_rfe_scaled = scaler.fit_transform(x_rfe)
    x_test_rfe_scaled = scaler.transform(x_test_rfe)
    x_target_rfe_scaled = scaler.transform(x_target_rfe)
    model_rfe, scores_rfe, grid_result_rfe, y_train_cross_validation_rfe = svr_model_linear(x_rfe_scaled, y_train)
    model_rfe.fit(x_rfe_scaled, y_train)
    y_train_predicted_rfe = model_rfe.predict(x_rfe_scaled)
    y_test_predicted_rfe = model_rfe.predict(x_test_rfe_scaled)
    y_target_predicted_rfe = model_rfe.predict(x_target_rfe_scaled)
    return model_rfe, grid_result_rfe, x_test_rfe, y_train_predicted_rfe, y_train_cross_validation_rfe, y_test_predicted_rfe, y_target_predicted_rfe


def summary_plot(y_train, y_test, y_train_predicted, y_train_cross_validation, y_test_predicted,
                 labelsize=12, legendsize=12, titlesize=14, textsize=12):
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 18))
    values = list(range(int(round(min(min(y_train),
                                      min(y_test),
                                      min(y_train_cross_validation),
                                      min(y_train_predicted),
                                      min(y_test_predicted)),
                                  0)),
                        int(round(max(max(y_train),
                                      max(y_test),
                                      max(y_train_cross_validation),
                                      max(y_train_predicted),
                                      max(y_test_predicted)),
                                  0))))

    rmse, mre, r2 = regression_metrics(y_train, y_train_predicted)
    plt.subplot(1, 3, 1)
    plt.text(0.05,
             0.625,
             'Median Error = ' + str(round(mre, 3)) + '%' + '\n' +
             'RMSE = ' + str(round(rmse, 3)) + ' $\AA^2$' + '\n' +
             '$R^2$ = ' + str(round(r2, 3)),
             ha='left',
             va='top',
             transform=ax1.transAxes,
             fontsize=textsize)
    plt.scatter(y_train,
                y_train_predicted,
                color='powderblue',
                marker='o',
                edgecolor='black',
                label='Calibration',
                s=50)
    m_0, b_0 = np.polyfit(y_train, y_train_predicted, 1)
    plt.plot(np.asarray(values),
             np.asarray(values) * m_0 + b_0,
             color='powderblue',
             linestyle='-',
             label='Calibration Fit',
             linewidth=1.4)
    plt.plot(values,
             np.asarray(values),
             color='black',
             linestyle='--',
             label='1:1',
             linewidth=1)
    plt.title('Calibration Prediction', fontsize=titlesize)
    plt.ylabel('Predicted CCS ($\AA^2$)', fontsize=labelsize)
    plt.xlabel('Measured CCS ($\AA^2$)', fontsize=labelsize)
    plt.axis('sqaure')
    plt.legend(loc='lower right', prop={'size': legendsize})
    plt.tick_params(axis='y', direction='inout', length=4)
    plt.tick_params(axis='x', direction='inout', length=4)

    rmse, mre, r2 = regression_metrics(y_train, y_train_cross_validation)
    plt.subplot(1, 3, 2)
    plt.text(0.05,
             0.625,
             'Median Error = ' + str(round(mre, 3)) + '%' + '\n' +
             'RMSE = ' + str(round(rmse, 3)) + ' $\AA^2$' + '\n' +
             '$R^2$ = ' + str(round(r2, 3)),
             ha='left',
             va='top',
             transform=ax2.transAxes,
             fontsize=textsize)
    plt.scatter(y_train,
                y_train_cross_validation,
                color='thistle',
                marker='o',
                edgecolor='black',
                label='CV',
                s=50)
    m_1, b_1 = np.polyfit(y_train, y_train_cross_validation, 1)
    plt.plot(values,
             np.asarray(values) * m_1 + b_1,
             color='thistle',
             linestyle='-',
             label='CV Fit',
             linewidth=1.4)
    plt.plot(values,
             np.asarray(values),
             color='black',
             linestyle='--',
             label='1:1',
             linewidth=1)
    plt.title('Cross-Validation Prediction', fontsize=titlesize)
    plt.ylabel('Predicted CCS ($\AA^2$)', fontsize=labelsize)
    plt.xlabel('Measured CCS ($\AA^2$)', fontsize=labelsize)
    plt.axis('square')
    plt.legend(loc='lower right', prop={'size': legendsize})
    plt.tick_params(axis='y', direction='inout', length=4)
    plt.tick_params(axis='x', direction='inout', length=4)

    rmse, mre, r2 = regression_metrics(y_test, y_test_predicted)
    plt.subplot(1, 3, 3)
    plt.text(0.05,
             0.625,
             'Median Error = ' + str(round(mre, 3)) + '%' + '\n' +
             'RMSE = ' + str(round(rmse, 3)) + ' $\AA^2$' + '\n' +
             '$R^2$ = ' + str(round(r2, 3)),
             ha='left',
             va='top',
             transform=ax3.transAxes,
             fontsize=textsize)
    plt.scatter(y_test,
                y_test_predicted,
                color='bisque', marker='o',
                edgecolor='black',
                label='Test',
                s=50)
    m_2, b_2 = np.polyfit(y_test, y_test_predicted, 1)
    plt.plot(values,
             np.asarray(values) * m_2 + b_2,
             color='bisque',
             linestyle='-',
             label='Test Fit',
             linewidth=1.4)
    plt.plot(values,
             np.asarray(values),
             color='black',
             linestyle='-',
             label='1:1',
             linewidth=1)
    plt.title('Validation Prediction', fontsize=titlesize)
    plt.ylabel('Predicted CCS ($\AA^2$)', fontsize=labelsize)
    plt.xlabel('Measured CCS ($\AA^2$)', fontsize=labelsize)
    plt.axis('square')
    plt.legend(loc='lower right', prop={'size': legendsize})
    plt.tick_params(axis='y', direction='inout', length=4)
    plt.tick_params(axis='x', direction='inout', length=4)

    plt.show()
    return f
