import pickle
from ccsp2.arguments import *
from ccsp2.data_io import *
from ccsp2.model import *
from ccsp2.predict import *


def run_ccsp2(args):
    if args['output'] == '':
        args['output'] = os.getcwd()

    if args['workflow'] == 'all':
        run_model_workflow(args)
        args['model'] = os.path.join(args['output'], args['model_fname'] + '.ccsp2')
        run_predict_workflow(args)
    elif args['workflow'] == 'model':
        run_model_workflow(args)
    elif args['workflow'] == 'predict':
        run_predict_workflow(args)


if __name__ == '__main__':
    args = get_args()
    run_ccsp2(args)
