
import time
from classification_pipeline_stages import *


class ClassificationPipeline():

    def __init__(self, out_filepath, in_filepath):

        self.dataset = h5py.File(out_filepath)
        h5py_file = h5py.File(in_filepath)
        if 'rgbd_data' in h5py_file.keys():
            self._num_images = h5py_file['rgbd_data'].shape[0]
        elif 'image' in h5py_file.keys():
            self._num_images = h5py_file['image'].shape[0]
        elif 'rgbd' in h5py_file.keys():
            self._num_images = h5py_file['rgbd'].shape[0]
        else:
            self._num_images = h5py_file['images'].shape[0]
        self._pipeline_stages = []

    def add_stage(self, stage):
        self._pipeline_stages.append(stage)

    def run(self):

        for index in range(self._num_images):

            print
            print 'starting ' + str(index) + " of " + str(self._num_images)
            print

            for stage in self._pipeline_stages:
                start_time = time.time()

                #check if hd5f dataset has been created yet, and create it if need be
                if not stage.dataset_inited(self.dataset):
                    stage.init_dataset(self.dataset)

                #actually process the data
                stage.run(self.dataset, index)

                print str(stage) + ' took ' + str(time.time() - start_time) + ' seconds to complete.'


class GraspClassificationPipeline(ClassificationPipeline):

    def __init__(self, out_filepath, in_filepath, model_filepath, input_key):

        ClassificationPipeline.__init__(self, out_filepath, in_filepath)

        self.add_stage(CopyInRaw(in_filepath, in_key=input_key, out_key='rgbd_data'))
        self.add_stage(FeatureExtraction(model_filepath,
                 in_key='rgbd_data_normalized',
                 out_key='extracted_features',
                 use_float_64=False))
        self.add_stage(Classification(model_filepath))





