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
import sys, getopt

import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score,log_loss
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KDTree
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

import warnings
warnings.simplefilter('ignore', np.RankWarning)
warnings.simplefilter('ignore', np.RankWarning)


from track_clustering import *
from detector_geometry import *
from utils import * 
from merging_tracks import *


def features_Vmatrix(N1_df1, angle, idx, df, x, y, z, d, r, a, ns):
    '''
    Calculate the features. 
    The features are calculated between two hits (e.g. distance between the candidate hit and the last hit of the track) or on a single hit (e.g. x,y,z of a given hit)

    Return a np array with the features
    '''
    idx0, idx1 = idx[0],idx[-1]
    idx_1, idx_m2 = idx[1], idx[-2]
        
    a0 = a[idx0]
    a1 = a[idx1]
    r0 = r[idx0]
    r1 = r[idx1]
            
    da1 = a[idx[-1]] - a[idx[-2]]
    dr1 = r[idx[-1]] - r[idx[-2]]
        
    direction0 = np.arctan2(r[idx[1]] - r[idx[0]], a[idx[1]] - a[idx[0]] ) 
    direction1 = np.arctan2(r[idx[-1]] - r[idx[-2]], a[idx[-1]] - a[idx[-2]]) 
    direction21 = np.arctan2(r[idx[2]] - r[idx[1]],a[idx[2]] - a[idx[1]]) 
    direction23 = np.arctan2(r[idx[-2]] - r[idx[-3]],a[idx[-2]] - a[idx[-3]])
    direction0_a = np.arctan2(r[idx[2]] - r[idx[0]],a[idx[2]] - a[idx[0]]) 
    direction1_a = np.arctan2(r[idx[-1]] - r[idx[-3]],a[idx[-1]] - a[idx[-3]]) 
    direction_lf = np.arctan2(r[idx[-1]] - r[idx[0]],a[idx[-1]] - a[idx[0]]) 
            
    dr0ns = r0-r[ns]
    da0ns = a0-a[ns]

    features = np.stack([N1_df1[ns], np.array([angle]*len(ns)), mul_(N1_df1[idx0], ns), mul_(len(idx), ns), mul_(direction_lf, ns), df.arctan2[ns],
                r[ns]-r1, np.fabs(np.arctan2(r[ns]-r1,a[ns]-a1)-direction1), np.fabs(np.arctan2(r[ns]-r1,a[ns]-a1)-direction1_a),
                np.fabs(np.arctan2(r[ns]-r1,a[ns]-a1)-direction23), np.arctan2(r[ns]-r1,a[ns]-a1), [direction1]*len(ns),
                a[ns]-a1, mul_(direction23, ns), mul_(direction1_a, ns), np.fabs(np.arctan2(r[ns]-r1,a[ns]-a1)-direction_lf),
                np.fabs(np.arctan2(r[ns]-r[idx_m2],a[ns]-a[idx_m2])-direction1_a),r[ns]-r[idx_m2],a[ns]-a[idx_m2],
                r0-r[ns], np.fabs(np.arctan2(r0-r[ns],a0-a[ns])-direction0), np.fabs(np.arctan2(r0-r[ns],a0-a[ns])-direction0_a),
                np.fabs(np.arctan2(r0-r[ns],a0-a[ns])-direction21), np.arctan2(r0-r[ns],a0-a[ns]), mul_(direction0, ns),
                a0-a[ns], mul_(direction21, ns), mul_(direction0_a, ns), np.fabs(np.arctan2(r0-r[ns],a0-a[ns])-direction_lf),
                np.fabs(np.arctan2(r[idx_1]-r[ns],a[idx_1]-a[ns])-direction0_a),r[idx_1]-r[ns],a[idx_1]-a[ns],
                mul_(r[idx[-1]] - r[idx[0]], ns), mul_(r[idx[0]], ns), r[ns],
                mul_(N1_df1[idx[0]], ns), d[ns]-d[idx0], d[ns]-d[idx1], np.fabs(np.arctan2(d[ns]-d[idx1],a[ns]-a1)-direction1),
                np.fabs(np.arctan2(d[idx0]-d[ns],a0-a[ns])-direction0),
                (z[ns]-z[idx1])/(r[ns]-r[idx1]), (z[idx0]-z[ns])/(r[idx0]-r[ns]), np.sqrt((r0-r[ns])**2+(a0-a[ns])**2),
                np.sqrt((r[ns]-r[idx1])**2+(a[ns]-a[idx1])**2),
                mul_(np.sqrt(r[idx1]**2 + a[idx1]**2), ns), mul_(np.sqrt(r[idx0]**2 + a[idx0]**2), ns), np.sin(np.arctan2(r[ns],a[ns])),
                (x[ns]-x[idx1])/(r[ns]-r[idx1]), (x[idx0]-x[ns])/(r[idx0]-r[ns]), (y[ns]-y[idx1])/(r[ns]-r[idx1]), (y[idx0]-y[ns])/(r[idx0]-r[ns]),
                (d[ns]-d[idx1])/(r[ns]-r[idx1]), (d[idx0]-d[ns])/(r[idx0]-r[ns]), (d[ns]-d[idx_m2])/(r[ns]-r[idx_m2]), (d[idx_1]-d[ns])/(r[idx_1]-r[ns]),
                np.sin(a[ns]-a[idx1]), np.cos(a[ns]-a[idx1]), np.sin(a[idx0]-a[ns]), np.cos(a[idx0]-a[ns]), mul_(np.max(r[ns]-r1), ns), mul_(np.max(r0-r[ns]), ns),
                mul_(np.min(np.fabs(np.arctan2(r[ns]-r1,a[ns]-a1)-direction1)), ns), mul_(np.min(np.fabs(np.arctan2(r0-r[ns],a0-a[ns])-direction0)), ns),x[ns],mul_(x[idx0], ns),mul_(x[idx1], ns),y[ns],mul_(y[idx0], ns),mul_(y[idx1], ns),
                z[ns],mul_(z[idx0], ns),mul_(z[idx1], ns), np.sin(a[ns]), np.cos(a[ns]), mul_(np.sin(a[idx0]), ns),mul_(np.cos(a[idx0]), ns),
                mul_(da1/(np.sqrt(da1**2+dr1**2)+1e-6), ns), mul_(dr1/(np.sqrt(da1**2+dr1**2)+1e-6), ns), dr0ns/(np.sqrt(da0ns**2+dr0ns**2)+1e-6)], axis=1)
                
    return features.astype('float16')

def extend_supervised_create_dataset(submission, hits, truth, training_aperture=2, k_min=50, angle_delta=2, extend_start_of_track=True, extend_end_of_track=True):
    
    '''
	Create a dataset for extending the beginning and the end of a track. 

	train_aprture, k_min, angle_delta - hyperparameters that regulate the potential hits to be part of the track
    '''
    df = submission.merge(hits,  on=['hit_id'], how='left')
    df = df.assign(particle_id = truth.particle_id)
    df = df.assign(d = np.sqrt( df.x**2 + df.y**2 + df.z**2 ))
    df = df.assign(r = np.sqrt( df.x**2 + df.y**2))
    df = df.assign(arctan2 = np.arctan2(df.z, df.r))
    df = df.assign(a = np.arctan2(df.y,df.x))
    
    df_track_ids = np.array(df['track_id'].values)
    
    df_values = df[['x','y','z','d','r','a','hit_id']].values
    N1 = get_list_value_count(submission['track_id'])

    
    X_start = []
    y_start = []
    
    X_end = []
    y_end = []
    
    for angle in range(-180,180,1):
        print ('\r %f'%angle, end='',flush=True)
        df1 = df.loc[(df.arctan2>(angle-angle_delta)/180*np.pi) & (df.arctan2<(angle+angle_delta)/180*np.pi)]
        N1_df1 = N1[(df.arctan2>(angle-angle_delta)/180*np.pi) & (df.arctan2<(angle+angle_delta)/180*np.pi)]
        
        min_num_neighbours = len(df1)
        if min_num_neighbours<4: continue
        
        hit_ids = df1.hit_id.values
        particle_ids = df1.particle_id.values
        track_ids_hit = df1.track_id.values
                
        x,y,z = df1.as_matrix(columns=['x', 'y', 'z']).T
        r  = np.sqrt(x**2 + y**2)
        r  = r/1000
        d  = np.sqrt(x**2 + y**2 + z**2)
        a  = np.arctan2(y,x)
        tree = KDTree(np.column_stack([a,r]))
            
        track_ids = list(df1.track_id.unique())
        num_track_ids = len(track_ids)
        min_length=3
            
        for i in range(num_track_ids):
            p = track_ids[i]
            if p==0: continue
                        
            idx = np.where(df1.track_id==p)[0]
                                
            if len(idx)<min_length: continue
                    
            if angle>0:
                idx = idx[np.argsort( z[idx])]
            else:
                idx = idx[np.argsort(-z[idx])]
           
            particle_id_track = truth.loc[df.track_id==p, 'particle_id'].value_counts().index.tolist()[0]
                    
            idx0, idx1 = idx[0],idx[-1]
            a0 = a[idx0]
            a1 = a[idx1]
            r0 = r[idx0]
            r1 = r[idx1]
                
            direction0 = np.arctan2(r[idx[1]] - r[idx[0]], a[idx[1]] - a[idx[0]] ) 
            direction1 = np.arctan2(r[idx[-1]] - r[idx[-2]], a[idx[-1]] - a[idx[-2]]) 
                 
            ## START OF THE TRACK
            if extend_start_of_track:
                ns = tree.query([[a0,r0]], k=min(k_min,min_num_neighbours))
                ns = np.squeeze(np.array(ns, dtype='int')[1])

                direction = np.arctan2(r0-r[ns],a0-a[ns])
                ns = ns[(r0-r[ns]>0) &(np.fabs(direction-direction0)<training_aperture)]


                if len(ns) > 0:
                    ## Add the new train example to the dataset (X_start & y_start)
                    X_start.append(features_Vmatrix(N1_df1, angle, idx, df, x, y, z, d, r, a, ns))
                    y_start.extend(particle_ids[ns] == particle_id_track)

            ## END OF THE TRACK  
            if extend_end_of_track:
              
                ns = tree.query([[a1,r1]], k=min(k_min,min_num_neighbours))
                ns = np.squeeze(np.array(ns, dtype='int')[1])

                direction = np.arctan2(r[ns]-r1,a[ns]-a1)
                ns = ns[(r[ns]-r1>0) &(np.fabs(direction-direction1)<training_aperture)] 

                if len(ns) > 0:
                    ## Add the new train example to the dataset (X_start & y_start)
                    X_end.append(features_Vmatrix(N1_df1, angle, idx, df, x, y, z, d, r, a, ns))
                    y_end.extend(particle_ids[ns] == particle_id_track)
                
    if extend_start_of_track:
        X_start = np.concatenate(X_start, axis=0).astype('float16')
    
    if extend_end_of_track:
        X_end = np.concatenate(X_end, axis=0).astype('float16')
    
    return X_start, np.array(y_start, dtype='int8'), X_end, np.array(y_end, dtype='int8')





def extend_using_supervised(submission, hits, gbm_start_classifier, gbm_end_classifier, inference_aperture=3, k_min=100, angle_delta=2,
                              extend_start_of_track=True, extend_end_of_track=True, threshold_start=0.5, threshold_end=0.5):
    '''
	Extend the tracks of a submission, following the prediction given by the classifier gbm_start_classifier and gbm_end_classifier

	train_aprture, k_min, angle_delta - hyperparameters that regulate the potential hits to be part of the track
	threshold_start, threshold_end - minimum threshold to declare that a hit belong to a track
    '''
    df = submission.merge(hits,  on=['hit_id'], how='left')
    df = df.assign(d = np.sqrt( df.x**2 + df.y**2 + df.z**2 ))
    df = df.assign(r = np.sqrt( df.x**2 + df.y**2))
    df = df.assign(arctan2 = np.arctan2(df.z, df.r))
    df = df.assign(a = np.arctan2(df.y,df.x))

    df_track_ids = np.array(df['track_id'].values)
    
    df_values = df[['x','y','z','d','r','a','hit_id']].values
    N1 = get_list_value_count(submission['track_id'])
    
    for angle in range(-180,180,1):
        print ('\r %f'%angle, end='',flush=True)
        df1 = df.loc[(df.arctan2>(angle-angle_delta)/180*np.pi) & (df.arctan2<(angle+angle_delta)/180*np.pi)]
        N1_df1 = N1[(df.arctan2>(angle-angle_delta)/180*np.pi) & (df.arctan2<(angle+angle_delta)/180*np.pi)]
        
        min_num_neighbours = len(df1)
        if min_num_neighbours<4: continue
        
        hit_ids = df1.hit_id.values
        track_ids_hit = df1.track_id.values
                
        x,y,z = df1.as_matrix(columns=['x', 'y', 'z']).T
        r  = np.sqrt(x**2 + y**2)
        r  = r/1000
        d  = np.sqrt(x**2 + y**2 + z**2)
        a  = np.arctan2(y,x)
        tree = KDTree(np.column_stack([a,r]))
            
        track_ids = list(df1.track_id.unique())
        num_track_ids = len(track_ids)
        min_length=3
            
        to_predict_start_track = []
        to_predict_end_track = []
        potential_new_start_track = []
        potential_new_end_track = []
        
        for i in range(num_track_ids):
            p = track_ids[i]
            if p==0: continue
            
            idx = np.where(df1.track_id==p)[0]
                
            if len(idx)<min_length: continue
                    
            if angle>0:
                idx = idx[np.argsort( z[idx])]
            else:
                idx = idx[np.argsort(-z[idx])]
                    
                    
            a0 = a[idx[0]]
            a1 = a[idx[-1]]
            r0 = r[idx[0]]
            r1 = r[idx[-1]]
                
            direction0 = np.arctan2(r[idx[1]] - r[idx[0]], a[idx[1]] - a[idx[0]] ) 
            direction1 = np.arctan2(r[idx[-1]] - r[idx[-2]], a[idx[-1]] - a[idx[-2]]) 
                 
            ## START OF THE TRACK
            if extend_start_of_track:
                ns = tree.query([[a0,r0]], k=min(k_min,min_num_neighbours))
                ns = np.squeeze(np.array(ns, dtype='int')[1])

                direction = np.arctan2(r0-r[ns],a0-a[ns])
                ns = ns[(r0-r[ns]>0) &(np.fabs(direction-direction0)<inference_aperture)]

                if len(ns) > 0:
                    # Calculate the features for the given observation
                    to_predict_start_track.append(features_Vmatrix(N1_df1, angle, idx, df, x, y, z, d, r, a, ns))

                    for n in ns:
                        # Save the potential number of the track_id
                        potential_new_start_track.append([np.squeeze(hit_ids[n]) - 1, p])
            
            ## END OF THE TRACK
            if extend_end_of_track:
            
                ns = tree.query([[a1,r1]], k=min(k_min,min_num_neighbours))
                ns = np.squeeze(np.array(ns, dtype='int')[1])

                direction = np.arctan2(r[ns]-r1,a[ns]-a1)
                ns = ns[(r[ns]-r1>0) &(np.fabs(direction-direction1)<inference_aperture)] 

                if len(ns) > 0:
                    # Calculate the features for the given observation
                    to_predict_end_track.append(features_Vmatrix(N1_df1, angle, idx, df, x, y, z, d, r, a, ns))

                    for n in ns:
                        # Save the potential number of the track_id
                        potential_new_end_track.append([np.squeeze(hit_ids[n]) - 1, p])
                
        ## NB: for a computational time point of view, the predictions are done after gathering all the observation for a given angle

        ## Predict the hits at the start of the track
        if extend_start_of_track and (len(to_predict_start_track) > 0):
            ## Predictions..
            is_same_track = gbm_start_classifier.predict(np.nan_to_num(np.concatenate(to_predict_start_track, axis=0)), num_iteration=gbm_start_classifier.best_iteration)
            arg_sorted = np.argsort(is_same_track)
            
            ## The predictions that reach the threashold are extended to the track saved in potential_new_start_track
            for (t_pos, p_s), is_st in zip(np.array(potential_new_start_track)[arg_sorted],np.array(is_same_track)[arg_sorted]):
                if np.squeeze(is_st) >= threshold_start:
                    df_track_ids[t_pos] = p_s
                    
        ## Predict the hits at the end of the track
        if extend_end_of_track and (len(to_predict_end_track) > 0):
            ## Predictions..
            is_same_track = gbm_end_classifier.predict(np.nan_to_num(np.concatenate(to_predict_end_track, axis=0)), num_iteration=gbm_end_classifier.best_iteration)
            arg_sorted = np.argsort(is_same_track)
            
            ## The predictions that reach the threashold are extended to the track saved in potential_new_start_track
            for (t_pos, p_s), is_st in zip(np.array(potential_new_end_track)[arg_sorted],np.array(is_same_track)[arg_sorted]):
                if np.squeeze(is_st) >= threshold_end:
                    df_track_ids[t_pos] = p_s

                        
        df['track_id'] = df_track_ids
        
        
    df['track_id'] = df_track_ids
    df = df[['event_id', 'hit_id', 'track_id']]
    return df




def lightGBM_hit_classification(X_train, y_train, X_valid, y_valid, num_leaves=200,max_depth=-1, learning_rate=0.1, n_estimators=3000, subsample_for_bin=100000, 
                                is_unbalance=False, min_data_in_leaf=20, bagging_fraction=1.0, bagging_freq=0, max_bin=255, min_sum_hessian_in_leaf=0.001,
                                min_gain_to_split=0, lambda_l2=0, scale_pos_weight=1.0, print_info=True):
    '''
	classification lightGBM model. 
    '''

    p_time = PrintTime()
    
    d_train = lgb.Dataset(X_train, label=y_train, free_raw_data=True)
    d_valid = lgb.Dataset(X_valid, label=y_valid, free_raw_data=True)
    
    params = {}
    params['learning_rate'] = learning_rate
    params['boosting_type'] = 'gbdt'
    params['objective'] = 'binary'
    params['subsample_for_bin'] = subsample_for_bin
    params['num_leaves'] = num_leaves
    params['max_depth'] = max_depth
    params['is_unbalance'] = is_unbalance
    params['scale_pos_weight'] = scale_pos_weight
    params['min_data_in_leaf'] = min_data_in_leaf
    params['bagging_fraction'] = bagging_fraction
    params['bagging_freq'] = bagging_freq
    params['max_bin'] = max_bin
    params['min_sum_hessian_in_leaf'] = min_sum_hessian_in_leaf
    params['min_gain_to_split'] = min_gain_to_split 
    params['lambda_l2'] = lambda_l2
    
    gbm = lgb.train(params, d_train, n_estimators, valid_sets=[d_train,d_valid], early_stopping_rounds=50, verbose_eval=20,feval=custom_f1_score)
    #gbm.save_model('model.lgb', num_iteration=gbm.best_iteration)
    
    valid_predicted = gbm.predict(X_valid, num_iteration=gbm.best_iteration)

    if print_info:
        print('\nF1 Valid',f1_score(y_valid, np.round(valid_predicted)))
        print('Logloss Valid',log_loss(y_valid, valid_predicted))
        print(confusion_matrix(y_valid,np.round(valid_predicted)))
        print('--------------------------')
    
    train_predicted = gbm.predict(X_train, num_iteration=gbm.best_iteration)
    if print_info:
        print('\nF1 train',f1_score(y_train, np.round(train_predicted)))
        print(confusion_matrix(y_train,np.round(train_predicted)))
        print('Feature importances:', list(gbm.feature_importance()))

        print(p_time.get_timer(reset_timer=True))
    
    return gbm



def create_massive_dataset(classifier, train_path, training_aperture=2, angle_delta=2, k_min=50, skip=30, nevents=5, keep_only_percentage=50, 
    extend_start_of_track=True, extend_end_of_track=True):
    '''
	Create a dataset to be used for the classification of a hit to a given track. It produce a X and y data for both the start and end of a track
	classifier - layer detector classifier

	extend_start_of_track - boolean to declare the will to extend the start of the tracks
    extend_end_of_track - boolean to declare the will to extend the end of the tracks
    '''
    X_start_massive = []
    y_start_massive = []
    X_end_massive = []
    y_end_massive = []
    
    p_timer = PrintTime()

    for event_id, hits, cells, particles, truth in load_dataset(train_path, skip=skip, nevents=nevents):
        print('---------- Event :',event_id, '----------')
        
        '''
		Predict basic tracks for the event
        '''
        rz_scale=[1.3,1.4,0.94,0.273,0.01]
        shifting=[('z_shift',0)]
        lab = unroll_helix_clustering(hits, classifier, truth, shifting=shifting, func='hough', dz0=0.002, num_step=40, eps=0.008, additional_theta=[0], rz_scale=rz_scale, print_info=True, threshold_value_post=15)

        '''
		Create the start & end dataset for the tracks just inferred
        '''

        extended_subm = create_one_event_submission(0, hits, lab)
        X_train_start, y_train_start, X_train_end, y_train_end = extend_supervised_create_dataset(extended_subm, hits, truth, training_aperture=training_aperture, angle_delta=angle_delta, k_min=k_min, 
                                                                                extend_start_of_track=extend_start_of_track, extend_end_of_track=extend_end_of_track)
        X_start_massive.append(X_train_start)
        y_start_massive.append(y_train_start)
        
        X_end_massive.append(X_train_end)
        y_end_massive.append(y_train_end)

    print('\t',p_timer.get_timer())

    return np.concatenate(X_start_massive, axis=0), np.concatenate(y_start_massive, axis=0), np.concatenate(X_end_massive, axis=0), np.concatenate(y_end_massive, axis=0)



def create_supervised_track_extension_models(start_model_name, end_model_name, train_path, training_aperture=2, train_angle_delta=2, train_k_min=50, num_train_events=2, num_valid_events=1,
	min_data_in_leaf=800, num_leaves=3000):
	
	'''
	It creates two models for the start and end of a track. It predict 1 if a hit belong to a track, 0 otherwise
	
	train_aprture, k_min, angle_delta - hyperparameters that regulate which hits can be part of a track
	min_data_in_leaf, num_leaves - LightGBM hyperparameters

	num_train_events, num_valid_events - number of events used to create the dataset
	'''
	classifier = get_layer_classifier(train_path)

	'''
	Create the train & validation dataset for both the start and end of the track
	'''
	X_train_start, y_train_start, X_train_end, y_train_end = create_massive_dataset(classifier, train_path, training_aperture=training_aperture, angle_delta=train_angle_delta, k_min=train_k_min,
        skip=50, nevents=num_train_events, extend_start_of_track=True, extend_end_of_track=True)
	print('Train start shape:', X_train_start.shape, y_train_start.shape)
	print('Train end shape:', X_train_end.shape)

	X_valid_start, y_valid_start, X_valid_end, y_valid_end = create_massive_dataset(classifier, train_path, training_aperture=training_aperture, angle_delta=train_angle_delta, k_min=train_k_min,
        skip=0, nevents=num_valid_events, extend_start_of_track=True, extend_end_of_track=True)
	print('Valid start shape:', X_valid_start.shape)
	print('Valid end shape:', X_valid_end.shape)


	'''
	LightGBM models
	'''

	scale_pos_weight = len(y_train_start[y_train_start==0])/len(y_train_start[y_train_start==1])
	gbm_start = lightGBM_hit_classification(X_train_start, y_train_start, X_valid_start, y_valid_start, min_data_in_leaf=min_data_in_leaf, num_leaves=num_leaves, scale_pos_weight=scale_pos_weight)

	scale_pos_weight = len(y_train_end[y_train_end==0])/len(y_train_end[y_train_end==1])
	gbm_end = lightGBM_hit_classification(X_train_end, y_train_end, X_valid_end, y_valid_end, min_data_in_leaf=min_data_in_leaf, num_leaves=num_leaves, scale_pos_weight=scale_pos_weight)

	gbm_start.save_model(start_model_name, num_iteration=gbm_start.best_iteration)
	gbm_end.save_model(end_model_name, num_iteration=gbm_end.best_iteration)
