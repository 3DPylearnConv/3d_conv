
import os

#this is the root directory of the data folder
HOME_PATH = os.environ["HOME_PATH"] + '/'

#the training data directory
TRAINING_DATASET_DIR = HOME_PATH + 'data/training_data/'

#the data used at runtime
RUNTIME_DATASET_DIR = HOME_PATH + 'data/runtime_data/'

#the results of running the runtime data through the
#trained model
OUTPUT_DATASET_DIR = HOME_PATH + 'data/output_data/'

#this points to the locations of all the trained models
MODEL_DIR = HOME_PATH + 'models/'

#this is the directory where we get the model yaml file and hyper parameters from.
MODEL_TEMPLATE_DIR = HOME_PATH + 'model_templates/'

