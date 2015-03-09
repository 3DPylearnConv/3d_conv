
import os

#this is the root directory of the data folder
PYLEARN_DATA_PATH = os.environ["PYLEARN2_DATA_PATH"]

#this points to the location off all the raw rgbd images
#these were either from:
#1)gazebo i.e. graspit data
#2)download i.e saxena data
RAW_TRAINING_DATASET_DIR = PYLEARN_DATA_PATH + 'unprocessed_training_data/'
#RAW_TRAINING_DATASET_DIR = "/media/Elements/gdl_data/aggregated_post_gazebo_grasps/1421939837/"

#this is where datasets that have been processed go.
#these datasets have have patches extracted, data normalized, etc
PROCESSED_TRAINING_DATASET_DIR = PYLEARN_DATA_PATH + 'processed_training_data/'
#PROCESSED_TRAINING_DATASET_DIR = '/media/Elements/gdl_data/processed_training_data/'

#this points to the locations of all the trained models
MODEL_DIR = os.path.expanduser('~/grasp_deep_learning/pylearn_classifier_gdl/models/')

#this is the directory where we get the model yaml file and hyper parameters from.
MODEL_TEMPLATE_DIR = os.path.expanduser("~/grasp_deep_learning/pylearn_classifier_gdl/model_templates/")

#this is the output from running a model on a dataset
HEATMAPS_DATASET_DIR = PYLEARN_DATA_PATH + 'heatmaps/'

PRIORS_DIR = PYLEARN_DATA_PATH + 'grasp_priors/'

