import numpy as np
import pandas as pd
import sys, getopt

from supervised_track_extension import *

if __name__ == '__main__':

    train_path = 'data/train_100_events'

    training_aperture = 2
    train_angle_delta = 2
    train_k_min = 50

    num_train_events = 1
    num_valid_events = 1

    min_data_in_leaf = 800
    num_leaves = 3000

    start_model_name = 'gbm_start.lgb'
    end_model_name = 'gbm_end.lgb'

    try:
        opts, args = getopt.getopt(sys.argv[1:], "s:e:", ["num_train_events=","num_valid_events="])
        for opt, arg in opts:
            if opt == '-s':
                start_model_name = str(arg)
            elif opt == '-e':
                end_model_name = str(arg)
            elif opt == '--num_train_events':
                num_train_events = int(arg)
            elif opt == '--num_valid_events':
                num_valid_events = int(arg)

    except Exception:
        print('Wrong args, check them')

    ## Create the models for extend the start and the end of the tracks (NB: will be created one model for the start and one for the end)
    create_supervised_track_extension_models(start_model_name, end_model_name, train_path, training_aperture, train_angle_delta, train_k_min, num_train_events, num_valid_events)

	