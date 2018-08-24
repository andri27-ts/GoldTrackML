import numpy as np
import pandas as pd
from trackml.dataset import load_event, load_dataset
from trackml.score import score_event

from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import math

from utils import *


def choose_longest_track_by_layer(d0, d1, s0, s1, threshold_value=19):
    '''
    Choose the longest track between s0 and s1
    d0, d1 - the lenghts of the tracks in s0 and s1
    s1, s0 - tracks to merge
    '''
    d0 = np.array(d0)
    d1 = np.array(d1)
    s0 = np.array(s0)
    s1 = np.array(s1)

    d0[d0 > 19] = 0
    d1[d1 > 19] = 0
    s1[d0 > d1] = s0[d0 > d1] + max(s1)

    return s1


def number_hits_different_module(labels, vlm_predicted):
    '''
    Calculate for each hit, the number of hits in its track that belong to a different volume and layer id

    labels - list of track id for each hit
    vlm_predicted - predicted volume&layer for each hit
    '''

    real_N = {}
    assert(len(labels) == len(vlm_predicted))
    for idx, (l,vlm_p) in enumerate(zip(labels, vlm_predicted)):
        if l in real_N:
            real_N[l].add(vlm_p)
        else:
            real_N[l] = set([vlm_p])
                
    return [len(real_N[x]) for x in labels]

def multiple_tracks_merge_by_layer(lab_list, hits, classifier, truth=None, return_result=True):
    '''
    Merge N tracks by taking the one with a higher number of hits belonging to different volume-layers

    lab_list - list of predicted id track for a "hits"
    hits - "hits" file
    classifier - volume-layers classifier
    truth - "truth" file to calculate the score
    return_results - boolean to return the result or the merged tracks

    '''
    s_combo = lab_list[0]
    vlm_predicted = classifier.predict(hits[['x','y','z']]) 

    for lb in lab_list[1:]:
        ## Calculate the number of hits with different volume-layer in a track 
        N1 = number_hits_different_module(s_combo, vlm_predicted)
        N2 = number_hits_different_module(lb, vlm_predicted)
        
        ## Merge lb and s_combo 
        s_combo = choose_longest_track_by_layer(N2, N1, lb, s_combo, threshold_value=17)

    if return_result:
        return score_event(truth, create_one_event_submission(0, hits, s_combo))
    else:
        return s_combo
