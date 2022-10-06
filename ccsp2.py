from ccsp2.arguments import *
from ccsp2.data_io import *
from ccsp2.model import *
from ccsp2.predict import *


def run_ccsp2(args):
    if args['output'] == '':
        args['output'] = os.getcwd()

    if args['test'] != '':
        train_book, test_book = import_training_data(args['train'], test_book_path=args['test'])
    elif args['test'] == '':
        train_book, test_book = import_training_data(args['train'], split_percentage=args['split_percentage'])
    target_book = import_query_data(args['query'])
    train_input_type, train_input_errors = check_inputs(train_book, column_title='Input')
    test_input_type, test_input_errors = check_inputs(test_book, column_title='Input')
    target_input_type, target_input_errors = check_inputs(target_book, column_title='Input')

    if len(train_input_errors) + len(test_input_errors) + len(target_input_errors) > 0:
        if len(train_input_errors) > 0:
            print('Assuming the training input type:', train_input_type, '\n')
            print("The following training inputs returned errors:", '\n')
            for i in train_input_errors:
                print(i, '\n')
        if len(test_input_errors) > 0:
            print('Assuming the testing input type:', test_input_type, '\n')
            print("The following testing inputs returned errors:", '\n')
            for i in test_input_errors:
                print(i, '\n')
        if len(target_input_errors) > 0:
            print('Assuming the target input type:', target_input_type, '\n')
            print("The following target inputs returned errors:", '\n')
            for i in target_input_errors:
                print(i, '\n')
        print("Please correct these inputs and try again.")
        sys.exit()

    x_train, y_train, x_test, y_test, x_target = variable_assigner(train_book,
                                                                   test_book,
                                                                   target_book,
                                                                   train_input_type=train_input_type,
                                                                   test_input_type=test_input_type,
                                                                   target_input_type=target_input_type)
    initial_prediction = initial_ccs_prediction(x_train,
                                                y_train,
                                                x_test,
                                                y_test,
                                                x_target,
                                                outlier_removal=True,
                                                threshold=1000)
    rfecv = rfe_variable_selection(initial_prediction['x_train_scaled'],
                                   y_train,
                                   initial_prediction['grid_results'],
                                   plot=args['plot'])
    rfe_prediction = rfe_ccs_prediction(initial_prediction['x_train_clean'],
                                        y_train,
                                        initial_prediction['x_test_clean'],
                                        y_test,
                                        initial_prediction['x_target_clean'],
                                        rfecv)

    '''if args['plot']:
        summary_plot_all = summary_plot(y_train,
                                        y_test,
                                        initial_prediction['y_train_predicted'],
                                        initial_prediction['y_train_cross_validation'],
                                        initial_prediction['y_test_predicted'],
                                        labelsize=12,
                                        legendsize=12,
                                        titlesize=14,
                                        textsize=12)
        summary_plot_rfe = summary_plot(y_train,
                                        y_test,
                                        rfe_prediction['y_train_predicted_rfe'],
                                        rfe_prediction['y_train_cross_validation_rfe'],
                                        rfe_prediction['y_test_predicted_rfe'],
                                        labelsize=12,
                                        legendsize=12,
                                        titlesize=14,
                                        textsize=12)'''

    train_book_output, test_book_output, target_book_output = train_book.copy(), test_book.copy(), target_book.copy()
    train_book_output['Calibration CCS Prediction'] = initial_prediction['y_train_predicted']
    train_book_output['Cross-Validation CCS Prediction'] = initial_prediction['y_train_cross_validation']
    test_book_output['Validation CCS Prediction'] = initial_prediction['y_test_predicted']
    target_book_output['Target CCS Prediction'] = initial_prediction['y_target_predicted']
    train_book_output['Calibration CCS Prediction RFE VS'] = rfe_prediction['y_train_predicted_rfe']
    train_book_output['Cross-Validation CCS Prediction RFE VS'] = rfe_prediction['y_train_cross_validation_rfe']
    test_book_output['Validation CCS Prediction RFE VS'] = rfe_prediction['y_test_predicted_rfe']
    target_book_output['Target CCS Prediction RFE VS'] = rfe_prediction['y_target_predicted_rfe']

    train_book_output.to_csv(os.path.join(args['output'], 'train_book_output.csv'), index=True)
    test_book_output.to_csv(os.path.join(args['output'], 'test_book_output.csv'), index=True)
    target_book_output.to_csv(os.path.join(args['output'], 'target_book_output.csv'), index=True)

    if args['plot']:
        summary_plot_all = summary_plot(y_train,
                                        y_test,
                                        initial_prediction['y_train_predicted'],
                                        initial_prediction['y_train_cross_validation'],
                                        initial_prediction['y_test_predicted'],
                                        labelsize=12,
                                        legendsize=12,
                                        titlesize=14,
                                        textsize=12)
        prediction_plot_calibration = prediction_plot(y_train,
                                                      initial_prediction['y_train_predicted'],
                                                      train_book,
                                                      hover_column=['Compound Name'],
                                                      title_string="Calibration Prediction")
        prediction_plot_cross_validation = prediction_plot(y_train,
                                                           initial_prediction['y_train_cross_validation'],
                                                           train_book,
                                                           hover_column=['Compound Name'],
                                                           title_string="Cross-Validation Prediction")
        prediction_plot_validation = prediction_plot(y_test,
                                                     initial_prediction['y_test_predicted'],
                                                     test_book,
                                                     hover_column=['Compound Name'],
                                                     title_string="Validation Prediction")
        model_diagnostic_plot = model_diagnostics_plot(initial_prediction['x_test_clean'],
                                                       initial_prediction['model'])
        summary_plot_rfe = summary_plot(y_train,
                                        y_test,
                                        rfe_prediction['y_train_predicted_rfe'],
                                        rfe_prediction['y_train_cross_validation_rfe'],
                                        rfe_prediction['y_test_predicted_rfe'],
                                        labelsize=12,
                                        legendsize=12,
                                        titlesize=14,
                                        textsize=12)
        prediction_plot_calibration_rfe = prediction_plot(y_train,
                                                          rfe_prediction['y_train_predicted_rfe'],
                                                          train_book,
                                                          hover_column=['Compound Name'],
                                                          title_string="Calibration Prediction")
        prediction_plot_cross_validation_rfe = prediction_plot(y_train,
                                                               rfe_prediction['y_train_cross_validation_rfe'],
                                                               train_book,
                                                               hover_column=['Compound Name'],
                                                               title_string="Cross-Validation Prediction")
        prediction_plot_validation_rfe = prediction_plot(y_test,
                                                         rfe_prediction['y_test_predicted_rfe'],
                                                         test_book,
                                                         hover_column=['Compound Name'],
                                                         title_string="Validation Prediction")
        model_diagnostic_plot_rfe = model_diagnostics_plot(rfe_prediction['x_test_rfe'],
                                                           rfe_prediction['model_rfe'])

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


if __name__ == '__main__':
    args = get_args()
    run_ccsp2(args)
