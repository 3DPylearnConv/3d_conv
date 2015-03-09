#!/usr/bin/env python
import rospy
import os
import h5py
import time

import numpy as np

import paths

from classification_pipelines import BarrettGraspClassificationPipeline
from pylearn_classifier_gdl.srv import CalculateGraspsService, CalculateGraspsServiceResponse


def init_save_file(input_data_file, input_model_file):

    dataset_filepath = paths.HEATMAPS_DATASET_DIR + 'temp/' + input_data_file + '_' + input_model_file + '.h5'

    if os.path.exists(dataset_filepath):
        os.remove(dataset_filepath)

    save_dset = h5py.File(dataset_filepath)

    return save_dset, dataset_filepath


def init_rgbd_file(dataset_file):

    raw_rgbd_filepath = paths.RAW_TRAINING_DATASET_DIR + "temp/" + dataset_file + '.h5'

    if os.path.exists(raw_rgbd_filepath):
        os.remove(raw_rgbd_filepath)

    input_dset = h5py.File(raw_rgbd_filepath)

    return input_dset, raw_rgbd_filepath


class GraspServer:

    def __init__(self):

        #conv_model_name = "processed_2.0m_7vc_barrett_18_grasp_types_5_layer_170x170_1_12_12_41"
        #conv_model_name = "processed_out_condensed_5_layer_170x170_2_6_16_55"
        #conv_model_name = "random_5_layer_170x170_2_6_16_43"
        #self.conv_model_name = "processed_sweet_guava_5_layer_72x72_2_13_10_29"

        #this is just the all bottle
        #self.conv_model_name = "processed_man-2_18_16_46_condensed72_5_layer_72x72_small_dset_2_18_17_29"

        self.conv_model_name = 'processed_gazebo_contact_and_potential_grasps-2_11_18_20_8_grasp_types72_5_layer_72x72_2_20_17_59'

        conv_model_filepath = paths.MODEL_DIR + self.conv_model_name + "/cnn_model.pkl"

        dataset_file = str(int(round(time.time() * 1000)))

        self.input_dset, raw_rgbd_filepath = init_rgbd_file(dataset_file)
        self.input_dset.create_dataset("rgbd", (1, 480, 640, 4))

        self.save_dset, save_filepath = init_save_file(dataset_file, self.conv_model_name)

        rospy.logout(raw_rgbd_filepath)
        rospy.logout(save_filepath)

        self.pipeline = BarrettGraspClassificationPipeline(save_filepath, raw_rgbd_filepath, conv_model_filepath, input_key="rgbd")
        self.pipeline.run()

        self.service = rospy.Service('calculate_grasps_service', CalculateGraspsService, self.service_request_handler)

    def service_request_handler(self, request):
        rospy.loginfo("received request")

        rgbd = np.array(request.rgbd).reshape((480, 640, 4))
        self.input_dset["rgbd"][0] = rgbd

        self.pipeline.run()

        response = CalculateGraspsServiceResponse()
        response.heatmaps = self.save_dset['rescaled_heatmaps'][0].flatten()
        response.heatmap_dims = self.save_dset['rescaled_heatmaps'][0].shape
        response.model_name = self.conv_model_name

        rospy.loginfo("sent response")
        return response


if __name__ == "__main__":

    rospy.init_node('grasp_server_node')

    rospy.loginfo("starting grasp_server_node...")
    grasp_server = GraspServer()
    rospy.loginfo("Grasp Server Node ready for requests.")

    rospy.spin()