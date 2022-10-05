from ccsp2.arguments import *
from ccsp2.data_io import *
from ccsp2.model import *
from ccsp2.predict import *


def run_ccsp2(args):
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
                                   initial_prediction['grid_results'])
    rfe_prediction = rfe_ccs_prediction(initial_prediction['x_train_clean'],
                                        y_train,
                                        initial_prediction['x_test_clean'],
                                        y_test,
                                        initial_prediction['x_target_clean'],
                                        rfecv)


if __name__ == '__main__':
    args = get_args()
    run_ccsp2(args)
