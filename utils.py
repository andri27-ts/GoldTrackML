import numpy as np
import pandas as pd
from trackml.dataset import load_event, load_dataset
from trackml.score import score_event
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from tqdm import tqdm
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


def create_one_event_submission(event_id, hits, labels):
    '''
    Create a one event submission
    '''
    sub_data = np.column_stack(([event_id]*len(hits), hits.hit_id.values, labels))
    submission = pd.DataFrame(data=sub_data, columns=["event_id", "hit_id", "track_id"]).astype(int)
    return submission


def value_counts(val_list):
    ### Get a list and return a dictionary with the number of instance for every instance
    lab_dict = {}
    for l in val_list:
        if l in lab_dict:
            lab_dict[l] += 1
        else:
            lab_dict[l] = 1
    
    return lab_dict

def get_list_value_count(val_list):
    ### Get a list and return a list with the number of instances for every instance
    
    lab_list = value_counts(val_list)
    list_value_count = [lab_list[x] for x in val_list]
        
    assert(len(list_value_count) == len(val_list))
    
    return np.array(list_value_count)


current_milli_time = lambda: int(round(time.time() * 1000))

class PrintTime:
    def __init__(self):
        self.p_time = current_milli_time()
        
    def reset_timer(self):
        self.p_time = current_milli_time()
        
    def get_timer(self,reset_timer=False):
        timer_str = str(datetime.timedelta(milliseconds=current_milli_time()-self.p_time))
        if reset_timer:
            self.p_time = current_milli_time()
        return timer_str
    
    
def mul_(inp, arr):
    return np.array([inp]*len(arr))

def custom_f1_score(preds, train_data):
    '''
    Custom f1 score for LightGBM
    '''
    labels = train_data.get_label()
    return 'f1', f1_score(labels, np.round(preds)), True