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

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KDTree

import warnings
import time

import random

warnings.simplefilter('ignore', np.RankWarning)
warnings.filterwarnings("ignore")


from utils import *
from merging_tracks import *

def unroll_helix_clustering(hits, classifier, truth=[], func='', shifting=[], dz0=-0.0007, num_step=20, eps=0.0035, threshold_value=19, rz_scale=[], print_info=True,
                                   threshold_value_post=13, additional_theta=[0]):
    '''
    Hits clustering using DBSCAN. The hits will be "unrolled" rotating them from dz0 value to -dz0 value with a num_step regular intervals.
    For each angle, the z value of the hits will be shifted by a given amount. 

    NB: nested cycles over additiona_theta, shifting and finally theta


    shifting - list of tuple. the first element is the axis to shift ('z_shift' for shift on the z axis), the second is the value by which the hit will be rotated
    additional_theta - additional theta shift (e.g. instead of start at 0 and finish at dz0, the angle will be shifted all by additional_theta[i])
    '''
    x = hits['x'].values
    y = hits['y'].values
    z = hits['z'].values

    vlm_predicted = classifier.predict(hits[['x','y','z']]) 

    s_combo = []

    ## Cycle on each additional_theta value
    for add_th in additional_theta:
        theta = np.linspace((dz0/num_step)*add_th, dz0+(dz0/num_step)*add_th, num_step)
        theta = [x for t in zip(theta, -theta) for x in t]
        
        ## Cycle on each shift
        for shift in shifting:
            s1 = []
            N1 = []
            
            ## Cycle on each theta value
            for ii, dz in enumerate(tqdm(theta)):

                ## Z shift
                if shift[0] == 'z_shift':
                    dbscan = DBSCANClusterer_shift(theta=dz, eps=eps, min_samples=3, func=func, rz_scale=rz_scale, z_shifted_by=shift[1])
                else:
                    print('Information unknown')

                ## Clustering predictions
                labels = dbscan.predict(np.stack([x,y,z], axis=1))

                ## Merge (on-theta) the old tracks with the new ones
                if(ii==0):
                    s1 = labels
                    N1 = number_hits_different_module(s1, vlm_predicted)
                else:
                    N2 = number_hits_different_module(labels, vlm_predicted)

                    s1 = choose_longest_track_by_layer(N2, N1, labels, s1, threshold_value=threshold_value)
                    N1 = number_hits_different_module(s1, vlm_predicted)

            ## Merge (on-shift) the old tracks with the new ones
            if len(s_combo) > 0:
                N1 = number_hits_different_module(s_combo, vlm_predicted)
                N2 = number_hits_different_module(s1, vlm_predicted)

                s_combo = choose_longest_track_by_layer(N2, N1, s1, s_combo, threshold_value=threshold_value)

                if print_info:
                    print('\tGlobal score:', np.round(score_event(truth, create_one_event_submission(0, hits, s_combo)),4), 
                          '\tInter score:', np.round(score_event(truth, create_one_event_submission(0, hits, s1)),4),
                          '\t',shift, '\t',add_th)
            else:
                s_combo = s1

                if print_info:
                    print('\tInter score:', np.round(score_event(truth, create_one_event_submission(0, hits, s1)),4))

    if len(truth) != 0:
        submission = create_one_event_submission(0, hits, s_combo)
        score = score_event(truth, submission)
        if print_info:
            print('Train score:', np.round(score,4)) 
        
        return s_combo
        
    return s_combo


class DBSCANClusterer_shift(object):
    '''
    DBSCAN clustering
    '''

    def __init__(self, theta, eps, min_samples, grze_preprocess=False, rz_scales=[1, 1, 1], func='', rz_scale=[], z_shifted_by=0):
        self.eps = eps
        self.theta = theta
        self.min_samples = min_samples
        self.rz_scales = rz_scales
        self.func = func
        self.rz_scale = rz_scale
        self.z_shifted_by=z_shifted_by
        
    def _preprocess(self, hits):

        x = hits[:,0]
        y = hits[:,1]
        z = hits[:,2]
        
        z = z + self.z_shifted_by
                    
        rt = np.sqrt(x**2 + y**2)
        rtz = np.sqrt(x**2 + y**2 + z**2)
        a0 = np.arctan2(y,x)
              
        z1 = z/rt
        z2 = z/rtz

        if self.func == 'hough':
            a1 = a0 - np.nan_to_num(np.arccos(rt*self.theta))
        else:
            a1 = a0+self.theta*abs(rt)

        ss = StandardScaler()

        ## Features
        X = ss.fit_transform(np.stack([np.cos(a1), np.sin(a1), z1, z2, np.sqrt(a1**2+rt**2)], axis=1)) 
        X = np.multiply(X, self.rz_scale)   

        return X
    
    def predict(self, hits):
        X = self._preprocess(hits)

        cl = DBSCAN(eps=self.eps, min_samples=self.min_samples, algorithm='ball_tree', n_jobs=-1, leaf_size=5)
        labels = cl.fit_predict(X)     
                
        return labels


