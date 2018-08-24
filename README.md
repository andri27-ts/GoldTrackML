# TrackML Competition
This repository contains my approach that won a gold medal (11th place) in the [TrackML Particle Tracking Challenge](https://www.kaggle.com/c/trackml-particle-identification) on Kaggle.

![](https://storage.googleapis.com/kaggle-media/competitions/CERN/cern_graphic.png)

## Getting started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.
For any problem please contact me at andri27@hotmail.it

### Prerequisites
What things you need to install the software and the minimum hardware required. 

### Software
To run the code it's required Python 3.5.2 or newer.
The libraries required can be installed running 

```
pip install -r requirements.txt
git clone https://github.com/LAL/trackml-library
pip install --user --editable trackml-library
``` 


#### Hardware
To train the models are required at least 4GB of RAM.
The models has been trained in less than 2 hours on a low performance 4 cores CPU, whereas the test predictions (all the 125 events) has been done on a 48 cores high performance CPU in about 5 hours.


### Data Setup
Download 'test.zip' and 'train_sample.zip' from Kaggle and unzip the files in the *data* folder

## Train the model
To build the model run the following command

```
python train.py -s models/model_start.lgb -e models/model_end.lgb --num_train_events 10 --num_valid_events 2
```

Arguments:
```
-s <start_model_path> output model's path to extend the hits at the start of the track
-e <end_model_path>   output model's path to extend the hits at the end of the track

--num_train_event <number_events>  number of events to use for the train
--num_valid_event <number_events>  number of events to use for the validation
```

Note: in the repository are already available the models trained on 10 events: *models/gbm_start_10x.lgb* and *models/gbm_end_10x.lgb*

## Running the predictions
The predictions can be done using only unsupervised algorithm (it reach about 0.69 of score).. 

```
python predict.py -s models/model_start.lgb -e models/model_end.lgb --with_supervised 0 --predictions train
```

.. or both unsupervised and supervised (it reach about 0.76 of score)

```
python predict.py -s models/model_start.lgb -e models/model_end.lgb --with_supervised 1 --predictions train
```

Arguments:
```
-s <start_model_path> input model's path to extend the start of the track 
-e <end_model_path>   input model's path to extend the end of the track

--with_supervised <0 or 1>       With *0* use only unsupervised models. With *1* use unsupervised and supervised.
--predictions <train or test>    With *train*, it predict on train events. With *test* it predict on test events. 
```

## Directory structure

```
detector_geometry.py -> Contains the code to predict the volume, layer and module of a hit.
merging_tracks.py -> Contains functions to merge two or more tracks
predict.py -> Predict the tracks
supervised_track_extension.py -> Contains the functions needed to create the tracks extension models
track_clustering.py -> Code to cluster the hits unrolling them to different angles
train.py -> Train the models to extend the tracks
utils.py -> Contains additional functions
```
