
import os

from classification_pipelines import *
from utils import paths
from utils import choose

def init_save_file(input_data_file, input_model_file):

    dataset_filepath = paths.OUTPUT_DATASET_DIR + input_data_file[:-3] + '_' + input_model_file + '.h5'

    if os.path.exists(dataset_filepath):
        os.remove(dataset_filepath)

    h5py.File(dataset_filepath)

    return dataset_filepath


def main():

    #choose the model you would like to use
    conv_model_name = choose.choose_from(paths.MODEL_DIR)
    conv_model_filepath = paths.MODEL_DIR + conv_model_name + "/cnn_model.pkl"

    #choose the data you would like to run through the model
    dataset_file = choose.choose_from(paths.TRAINING_DATASET_DIR)
    raw_rgbd_filepath = paths.TRAINING_DATASET_DIR + dataset_file

    #choose where to save the output
    save_filepath = init_save_file(dataset_file, conv_model_name)

    pipelines = [("grasp", GraspClassificationPipeline)]

    pipeline = choose.choose(pipelines, 'pipeline')

    pipeline = pipeline(save_filepath, raw_rgbd_filepath, conv_model_filepath, input_key="rgbd")

    pipeline.run()

if __name__ == "__main__":
    main()
