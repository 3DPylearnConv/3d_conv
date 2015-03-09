
import os
import shutil
import time
import h5py

from pylearn2.testing import skip
from pylearn2.config import yaml_parse

import paths
import choose

#helper to build up the save path from the name of the model and the current time.
def get_save_path(model_template, dataset):
    t = time.localtime()

    minute = str(t.tm_min)
    if len(minute) == 1:
        minute = '0' + minute

    t_string = str(t.tm_mon) + "_" + str(t.tm_mday) + "_" + str(t.tm_hour) + "_" + minute

    #remove '.h5' from the dataset
    dataset = dataset[:-3]

    return paths.MODEL_DIR + dataset + "_" + model_template + "_" + t_string


#have the user choose:
# 1) a model template
# 2) a dataset to train against.
#then modify the hyper_params to specify a save location as well as the dataset used.
def build_model():

    dataset_filename = choose.choose_from(paths.PROCESSED_TRAINING_DATASET_DIR)
    model_template = choose.choose_from(paths.MODEL_TEMPLATE_DIR)

    model_template_yaml = open(paths.MODEL_TEMPLATE_DIR + model_template + "/model.yaml", 'r').read()
    hyper_params_file = open(paths.MODEL_TEMPLATE_DIR + model_template + "/hyper_params.yaml", 'r').read()

    dataset = h5py.File(paths.PROCESSED_TRAINING_DATASET_DIR + dataset_filename)
    num_labels = dataset['c01b_train_patch_labels'].shape[-1]

    hyper_params_dict = yaml_parse.load(hyper_params_file)
    hyper_params_dict['save_path'] = get_save_path(model_template, dataset_filename)
    hyper_params_dict['dataset'] = paths.PROCESSED_TRAINING_DATASET_DIR + dataset_filename
    hyper_params_dict['num_output_channels'] = num_labels

    return model_template_yaml, hyper_params_dict


#we want to remove the directory for the save model if it exists, then we want to
#save the hyper params and model_yaml that were used.  This way, we have a copy of them
#even if the template files are changed.
def prep_model_save_path(save_path, model_yaml, hyper_params_dict):

    assert paths.MODEL_DIR in save_path

    if os.path.exists(save_path):
        shutil.rmtree(save_path)

    os.mkdir(save_path)

    f = open(save_path + '/' + 'model.yaml', 'w')
    f.write(model_yaml)
    f.close()

    f = open(save_path + '/' + 'hyper_params.yaml', 'w')
    for key in hyper_params_dict.keys():
        f.write(str(key) + ': ' + str(hyper_params_dict[key]) + '\n')
    f.close()


#this method verifies that we have data, and actually begins training the model
def train_convolutional_network(model_with_hyper_params):
    skip.skip_if_no_data()
    train = yaml_parse.load(model_with_hyper_params)
    train.main_loop()


def main():
    #ask user for input in order to get the model and hyper parameters we want to use
    model_template_yaml, hyper_params_dict = build_model()

    #save the model and hyper params in the save dir so that we can access them
    #whenever we want, even if the template files have been changed.
    prep_model_save_path(hyper_params_dict['save_path'], model_template_yaml, hyper_params_dict)

    #insert the hyper parameters into the model template
    model = model_template_yaml % hyper_params_dict

    #train the model.
    train_convolutional_network(model)

if __name__ == "__main__":
    main()


