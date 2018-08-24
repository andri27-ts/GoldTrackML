from sklearn.neighbors import KNeighborsClassifier

import numpy as np
import pandas as pd

from trackml.dataset import load_event, load_dataset
from trackml.score import score_event

def acquire_vlm_dataset(skip, nevents, train_path):
    '''
    Create the dataset to predict the volume and layer id
    '''
    X_dataset = []
    y_dataset = []
    for event_id, hits, cells, particles, truth in load_dataset(train_path, skip=skip, nevents=nevents):
        
        vlm = (hits.volume_id *100) + (hits.layer_id) 
        X_dataset.append(hits[['x','y','z']].values)
        y_dataset.append(vlm)
        
    return np.concatenate(X_dataset, axis=0), np.concatenate(y_dataset, axis=0)

def acquire_vlm_module_dataset(skip, nevents, train_path):
    '''
    Create the dataset to predict the module id
    '''
    X_dataset = []
    y_dataset = []
    for event_id, hits, cells, particles, truth in load_dataset(train_path, skip=skip, nevents=nevents):
        vlm = hits.module_id
        X_dataset.append(hits[['x','y','z']].values)
        y_dataset.append(vlm)
        
    return np.concatenate(X_dataset, axis=0), np.concatenate(y_dataset, axis=0)

def get_layer_classifier(train_path):
    '''
    Return a classifier to predict the volume and layer id of a hit
    '''

    # Create the dataset
    X,y = acquire_vlm_dataset(50,10, train_path)
    # Create the model
    classifier = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
    classifier.fit(X, y)

    return classifier

    
