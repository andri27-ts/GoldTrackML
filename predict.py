from multiprocessing import Pool
from trackml.dataset import load_event, load_dataset
from trackml.score import score_event
from sklearn.neighbors import KNeighborsClassifier

import numpy as np
import pandas as pd
from trackml.dataset import load_event, load_dataset
from trackml.score import score_event
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from tqdm import tqdm, tqdm_notebook
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import datetime
import random
from sys import getsizeof

import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import sys, getopt

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score,log_loss
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KDTree
from sklearn.metrics import classification_report, confusion_matrix

import warnings
warnings.simplefilter('ignore', np.RankWarning)
warnings.simplefilter('ignore', np.RankWarning)

from track_clustering import *
from detector_geometry import *
from utils import * 
from merging_tracks import *
from supervised_track_extension import extend_using_supervised



def DBSCAN_clustering(hits, classifier):
    '''
    Cluster the hits with the same track using DBSCAN algorithm

    hits - hits of a event
    classifier - classifier to predict the layer of every hit

    return for every hit, the identifier of the track it belong to
    '''
    
    '''rz_scale=[1.3,1.4,0.94,0.273,0.01]

    shifting=[('z_shift',6), ('z_shift',-6), ('z_shift',2), ('z_shift',-2)]
    lab1 = unroll_helix_clustering(hits, classifier, shifting=shifting, func='hough', dz0=0.002, num_step=40, eps=0.0085, additional_theta=[0], rz_scale=rz_scale, print_info=False, threshold_value_post=15)

    lab2 = unroll_helix_clustering(hits, classifier, shifting=shifting, func='hough', dz0=0.002, num_step=40, eps=0.0085, additional_theta=[1/2], rz_scale=rz_scale, print_info=False, threshold_value_post=15)

    shifting=[('z_shift',5), ('z_shift',-5), ('z_shift',1.5), ('z_shift',-1.5)]
    lab3 = unroll_helix_clustering(hits, classifier, shifting=shifting, func='hough', dz0=0.002, num_step=40, eps=0.008, additional_theta=[1/4], rz_scale=rz_scale, print_info=False, threshold_value_post=15)

    shifting=[('z_shift',4), ('z_shift',-4), ('z_shift',1), ('z_shift',-1)]
    lab4 = unroll_helix_clustering(hits, classifier, shifting=shifting, func='hough', dz0=0.002, num_step=40, eps=0.008, additional_theta=[3/4],rz_scale=rz_scale, print_info=False, threshold_value_post=15)

    # Merge the tracks by choosing the one with the higher number of intersecting layers
    labs = multiple_tracks_merge_by_layer([lab4, lab3, lab2, lab1], hits, classifier, return_result=False)
    '''
    
    
    rz_scale=[1.3,1.4,0.94,0.273,0.01]
    
    shifting=[('z_shift',2), ('z_shift',-2), ('z_shift',6), ('z_shift',-6)]
    lab1 = unroll_helix_clustering(hits, classifier, shifting=shifting, func='hough', dz0=0.002, num_step=60, eps=0.008, additional_theta=[0], rz_scale=rz_scale, print_info=False, threshold_value_post=15)

    shifting=[('z_shift',2.1), ('z_shift',-2.1), ('z_shift',6.1), ('z_shift',-6.1)]
    lab2 = unroll_helix_clustering(hits, classifier, shifting=shifting, func='hough', dz0=0.002, num_step=60, eps=0.008, additional_theta=[1/2], rz_scale=rz_scale, print_info=False, threshold_value_post=15)

    shifting=[('z_shift',3), ('z_shift',-3), ('z_shift',7), ('z_shift',-7)]
    lab3 = unroll_helix_clustering(hits, classifier, shifting=shifting, func='hough', dz0=0.002, num_step=60, eps=0.008, additional_theta=[3/4], rz_scale=rz_scale, print_info=False, threshold_value_post=15)

    shifting=[('z_shift',2), ('z_shift',-2), ('z_shift',8), ('z_shift',-8)]
    lab4 = unroll_helix_clustering(hits, classifier, shifting=shifting, func='hough', dz0=0.002, num_step=60, eps=0.008, additional_theta=[1/4], rz_scale=rz_scale, print_info=False, threshold_value_post=15)
    
    shifting=[('z_shift',1.5), ('z_shift',-1.5), ('z_shift',5), ('z_shift',-5)]
    lab5 = unroll_helix_clustering(hits, classifier, shifting=shifting, func='hough', dz0=0.002, num_step=60, eps=0.0075, additional_theta=[1/8], rz_scale=rz_scale, print_info=False, threshold_value_post=15)

    shifting=[('z_shift',1), ('z_shift',-1), ('z_shift',4), ('z_shift',-4)]
    lab6 = unroll_helix_clustering(hits, classifier, shifting=shifting, func='hough', dz0=0.002, num_step=60, eps=0.0075, additional_theta=[3/8], rz_scale=rz_scale, print_info=False, threshold_value_post=15)

    shifting=[('z_shift',0), ('z_shift',3), ('z_shift',-3)]
    lab7 = unroll_helix_clustering(hits, classifier, shifting=shifting, func='hough', dz0=0.002, num_step=60, eps=0.0075, additional_theta=[5/8], rz_scale=rz_scale, print_info=False, threshold_value_post=15)

    shifting=[('z_shift',1), ('z_shift',-1), ('z_shift',6), ('z_shift',-6)]
    lab8 = unroll_helix_clustering(hits, classifier, shifting=shifting, func='hough', dz0=0.002, num_step=60, eps=0.007, additional_theta=[7/8], rz_scale=rz_scale, print_info=False, threshold_value_post=15)
    
    shifting=[('z_shift',0), ('z_shift',5), ('z_shift',-5)]
    lab9 = unroll_helix_clustering(hits, classifier, shifting=shifting, func='hough', dz0=0.002, num_step=60, eps=0.007, additional_theta=[1/16], rz_scale=rz_scale, print_info=False, threshold_value_post=15)
    
    shifting=[('z_shift',2.5), ('z_shift',-2.5), ('z_shift',6.5), ('z_shift',-6.5)]
    lab10 = unroll_helix_clustering(hits, classifier, shifting=shifting, func='hough', dz0=0.002, num_step=60, eps=0.0065, additional_theta=[7/16], rz_scale=rz_scale, print_info=False, threshold_value_post=15)

    shifting=[('z_shift',3.2), ('z_shift',-3.2), ('z_shift',7.2), ('z_shift',-7.2)]
    lab11 = unroll_helix_clustering(hits, classifier, shifting=shifting, func='hough', dz0=0.0021, num_step=60, eps=0.006, additional_theta=[13/16], rz_scale=rz_scale, print_info=False, threshold_value_post=15)
    
 
    # Merge the tracks by choosing the one with the higher number of intersecting layers
    labs = multiple_tracks_merge_by_layer([lab11, lab10, lab9, lab8, lab7, lab6, lab5, lab4, lab3, lab2, lab1, lab1, lab2, lab3, lab4, lab5, lab6, lab7, lab8, lab9, lab10, lab11], hits, classifier, return_result=False)
    
    return labs

def clustering_and_extension(event_id, hits, classifier, start_model_name, end_model_name, with_supervised):
    
    '''
    Cluster the hits and expand the obtained tracks using a supervised algorithm

    event_id - Event Id
    hits - hits of an event
    classifier - classifier to predict the layer of every hit 
    start_model_name - name of the model that extend the initial part of the track
    end_model_name - name of the model that extend the last part of the track

    return for every hit, the identifier of the track it belong to
    '''
    

    p_timer = PrintTime()

    ## CLUSTERING - Cluster the hits that belong to the same track (use DBSCAN)
    labs_ref = DBSCAN_clustering(hits, classifier)

    print('\t DBSCAN Time:', p_timer.get_timer())
    
    submission = create_one_event_submission(event_id, hits, labs_ref)

    ## TRACK EXTENSION - extend the start and the end of every track using a supervised technique
    if with_supervised:
        p_timer = PrintTime()

        threshold_start = 0.7
        threshold_end = 0.7
        # load the models to extend the "start" and "end" of the track
        lgbm_start = lgb.Booster(model_file=start_model_name, silent=True)
        lgbm_end = lgb.Booster(model_file=end_model_name, silent=True)
        
        iter_ext_subm = create_one_event_submission(0, hits, labs_ref)

        # iterate the extension algorithm 2 times (with 3-4 times it reach the best performance)
        for i in range(2):
            iter_ext_subm = extend_using_supervised(iter_ext_subm, hits, lgbm_start, lgbm_end, inference_aperture=2, angle_delta=2, k_min=100, 
                                                    threshold_start=threshold_start, threshold_end=threshold_end, extend_start_of_track=True, extend_end_of_track=True)
        submission = create_one_event_submission(event_id, hits, iter_ext_subm['track_id'].values)

        print('\t EXTEND Time:', p_timer.get_timer())

    return submission




def worker(event_id, hits, cells, classifier, start_model_name, end_model_name, with_supervised):
	print('Event ID: ', event_id)

	sub = clustering_and_extension(event_id, hits, classifier, start_model_name, end_model_name, with_supervised)
	return sub


path_to_test = 'data/test' 
train_path = 'data/train_100_events' 

if __name__ == '__main__':
    print('Starting main..')
    predict_test = False
    n_proc = 4
    start_hits_classifier = 'gbm_start_10x.lgb' 
    end_hits_classifier = 'gbm_end_10x.lgb' 

    skip=30
    nevents=2

    with_supervised = True
    predictions = 'train'

    try:
        opts, args = getopt.getopt(sys.argv[1:], "s:e:", ["with_supervised=","predictions="])
        for opt, arg in opts:
            if opt == '-s':
                start_hits_classifier = str(arg)
            elif opt == '-e':
                end_hits_classifier = str(arg)
            elif opt == '--with_supervised':
                with_supervised = bool(int(arg))
            elif opt == '--predictions':
                predictions = str(arg)

    except Exception:
        print('Wrong args, check them!')

    classifier = get_layer_classifier(train_path)

    if predictions == 'test':
        print('Test predictions..')
        ## Use multi-processes to predict the tracks on the test events
        pool = Pool(processes=n_proc, maxtasksperchild=1)
        test_predictions = pool.starmap(worker,[(i,h,c,classifier,start_hits_classifier,end_hits_classifier, with_supervised) for i,h,c in load_dataset(path_to_test, parts=['hits', 'cells'])])
        pool.close()

        ## Save the predictions
        submission = pd.concat(test_predictions, axis=0)
        submission['track_id'] = submission['track_id'] + 1
        submission.to_csv('submission_01.csv', index=False)

    elif predictions == 'train':
        ## Use multi-processes to predict the tracks on the train events
        print('Train predictions..')
        pool = Pool(processes=n_proc, maxtasksperchild=1)
        train_predictions = pool.starmap(worker, [(i,h,c, classifier, start_hits_classifier, end_hits_classifier, with_supervised) for i,h,c,p,t in load_dataset(train_path, skip=skip, nevents=nevents)])
    
        ## Calulate the score for each event
        events_score = []
        for sub, (i,h,c,p,truth) in zip(train_predictions, load_dataset(train_path, skip=skip, nevents=nevents)):

            score = score_event(truth, sub)
            print('Event',i,'->',score)
            events_score.append(score)
    
        ## Calculate the mean of the scores
        print('Mean score:', np.mean(events_score))
    else:
        print('Predictions argument wrong!')
