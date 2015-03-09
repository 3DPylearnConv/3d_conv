
import h5py
import numpy as np
import math
import os
import random

from pylearn2.datasets import preprocessing
from pylearn2.expr.preprocessing import global_contrast_normalize
import matplotlib.pyplot as plt

from subtractive_divisive_lcn import *



class ExtractRawGraspData(preprocessing.Preprocessor):

    def __init__(self, raw_data_folder,  data_labels=("rgbd_patches", "patch_labels")):
        self.raw_data_folder = raw_data_folder
        self.data_labels = data_labels

    def apply(self, dataset, can_fit=False):
        print self

        #check if we have already extracted the raw data
        if self.data_labels[0] in dataset.keys() or self.data_labels[1] in dataset.keys():
            print "skipping extract_raw_data, this has already been run"
            return

        dataset.create_dataset(self.data_labels[0], (0, 480, 640, 4), chunks=(100, 480, 640, 4), maxshape=(None, 480, 640, 4))
        dataset.create_dataset(self.data_labels[1], (0, 480, 640), chunks=(100, 480, 640), maxshape=(None, 480, 640))

        current_total = 0
        for subdirectory in os.walk(self.raw_data_folder).next()[1]:

            print subdirectory

            old_dataset = h5py.File(self.raw_data_folder + subdirectory + '/rgbd_and_labels.h5')
            number_to_add = old_dataset['rgbd'].shape[0]

            for label in self.data_labels:
                new_size = list(dataset[label].shape)
                new_size[0] += number_to_add
                new_size = tuple(new_size)
                dataset[label].resize(new_size)
                dataset[label][current_total: current_total+number_to_add] = old_dataset[label][:]

            current_total += number_to_add



class CopyInRaw(preprocessing.Preprocessor):

    def __init__(self, source_dataset_filepath, input_keys, output_keys):
        self.source_dataset = h5py.File(source_dataset_filepath)
        self.input_keys = input_keys
        self.output_keys = output_keys

    def apply(self, dataset, can_fit=False):
        print self
        for index in range(len(self.input_keys)):
            input_key = self.input_keys[index]
            output_key = self.output_keys[index]
            shape = self.source_dataset[input_key].shape
            dataset.create_dataset(output_key, shape)

            for i in range(shape[0]):
                if i % 1000 == 0:
                    print str(i) + ' / ' + str(shape[0])
                dataset[output_key][i] = self.source_dataset[input_key][i]



class RandomizePatches(preprocessing.Preprocessor):

    def __init__(self, keys):
        self.keys = keys

    def apply(self, dataset, can_fit=False):
        print self
        label_key, patch_key = self.keys
        num_images = dataset[label_key].shape[0]

        for i in range(num_images):
            rand_index = random.randint(0, dataset[label_key].shape[0]-1)

            rand_patch = np.copy(dataset[patch_key][rand_index])
            rand_label = np.copy(dataset[label_key][rand_index])

            index_patch = np.copy(dataset[patch_key][i])
            index_label = np.copy(dataset[label_key][i])

            dataset[patch_key][i] = rand_patch
            dataset[label_key][i] = rand_label

            dataset[patch_key][rand_index] = index_patch
            dataset[label_key][rand_index] = index_label


class NormalizePatches(preprocessing.Preprocessor):

    def __init__(self, keys):
        self.keys = keys

    def apply(self, dataset, can_fit=False):
        print self
        for key in self.keys:
            num_images = dataset[key].shape[0]
            num_channels = dataset[key].shape[-1]

            for i in range(num_images):
                if i % num_images/10 == 0:
                    print str(i) + ' / ' + str(num_images)
                for j in range(num_channels):
                    dataset[key][i, :, :, j] = dataset[key][i, :, :, j] / dataset[key][i, :, :, j].max()


class LecunSubtractiveDivisiveLCN(preprocessing.Preprocessor):

    def __init__(self, in_key, out_key):
        self.in_key = in_key
        self.out_key = out_key
        self.sub_div_fcn = None

    def apply(self, dataset, can_fit=False):
        print self
        num_images = dataset[self.in_key].shape[0]
        shape = dataset[self.in_key].shape
        dataset.create_dataset(self.out_key, shape, chunks=tuple([10] + list(shape[1:])))

        for index in range(num_images):
            if index % (num_images/10) == 0:
                print str(index) + ' / ' + str(num_images)

            img = dataset[self.in_key][index]
            num_channels = img.shape[-1]

            img_out = np.zeros_like(img)

            img_in = np.zeros((1, img.shape[0], img.shape[1]), dtype=np.float32)
            if not self.sub_div_fcn:
                self.sub_div_fcn = subtractive_divisive_lcn(img_in, img_shape=img.shape[0:2], kernel_shape=9)

            for i in range(num_channels):
                img_in[0] = img[:, :, i]
                img_out[:, :, i] = self.sub_div_fcn(img_in.reshape((img_in.shape[0], img_in.shape[1], img_in.shape[2], 1)))

            dataset[self.out_key][index] = img_out


class SplitGraspPatches(preprocessing.Preprocessor):

    def __init__(self,
                 output_keys=(("train_patches", "train_patch_labels"), ("valid_patches", "valid_patch_labels"), ("test_patches", "test_patch_labels")),
                 output_weights = (.8, .1, .1),
                 source_keys=("rgbd_patches", "rgbd_patch_labels")):

        self.output_keys = output_keys
        #normalize the output weights
        self.output_weights = [x/sum(output_weights) for x in output_weights]
        self.source_keys = source_keys

    def apply(self, dataset, can_fit=False):
        print self
        #check if we have already extracted patches for this set of patch_labels
        # if self.output_keys[0][0] in dataset.keys():
        #     print "skipping split_patches, this has already been run"
        #     return



        for index in range(len(self.output_keys)):
            output_key_pair = self.output_keys[index]
            patch_key = output_key_pair[0]
            label_key = output_key_pair[1]

            num_patches = math.floor(self.output_weights[index] * dataset[self.source_keys[0]].shape[0])
            num_patches = num_patches - (num_patches % 20)
            patch_shape = dataset[self.source_keys[0]].shape[1:4]
            num_labels = dataset[self.source_keys[1]].shape[-1]

            start_range = math.floor(sum(self.output_weights[:index])*dataset[self.source_keys[0]].shape[0])
            #import IPython
            #IPython.embed()
            dataset.create_dataset(patch_key, (int(num_patches), patch_shape[0], patch_shape[1], patch_shape[2]))
            dataset.create_dataset(label_key, (int(num_patches), num_labels))

            for i in range(int(num_patches)):

                dataset[patch_key][i] = dataset[self.source_keys[0]][int(start_range) + i]
                dataset[label_key][i] = dataset[self.source_keys[1]][int(start_range) + i]


class ExtractGraspPatches(preprocessing.Preprocessor):

    def __init__(self,
                 patch_shape=(25, 25),
                 patch_labels=("rgbd_patches", "patch_labels"),
                 patch_source_labels=("rgbd", "labels"),
                 num_patches=100000):

        self.patch_shape = patch_shape
        self.patch_labels = patch_labels
        self.patch_source_labels = patch_source_labels
        self.num_patches = num_patches

    def apply(self, dataset, can_fit=False):
        #check if we have already extracted patches for this set of patch_labels
        if self.patch_labels[0] in dataset.keys() or self.patch_labels[1] in dataset.keys():
            print "skipping extract_patches, this has already been run"
            return

        dataset.create_dataset(self.patch_labels[0], (self.num_patches, self.patch_shape[0], self.patch_shape[1], 4), chunks=(100, self.patch_shape[0], self.patch_shape[1], 4))
        dataset.create_dataset(self.patch_labels[1], (self.num_patches, 1), chunks=(100, 1))


        X = dataset[self.patch_source_labels[0]]
        y = dataset[self.patch_source_labels[1]]

        num_images = X.shape[0]

        patch_X = dataset[self.patch_labels[0]]
        patch_y = dataset[self.patch_labels[1]]

        channel_slice = slice(0, X.shape[-1])

        rng = preprocessing.make_np_rng([1, 2, 3], which_method="randint")

        i = 0
        iteration_count = 0
        while i < self.num_patches:
            if iteration_count % 1000 == 0:
                print "extracting patches: " + str(i) + "/" + str(self.num_patches)

            image_num = rng.randint(num_images)
            x_args = [image_num]
            y_args = [image_num]

            for x_index in range(y.shape[1]):
                for y_index in range(y.shape[2]):
                    if x_index < y.shape[1]-self.patch_shape[0] and y_index < y.shape[2]-self.patch_shape[1]:
                        if y[image_num, x_index, y_index] > 0:

                            x_args.append(slice(x_index, x_index + self.patch_shape[0]))
                            y_args.append(x_index + self.patch_shape[0]/2.0)

                            x_args.append(slice(y_index, y_index + self.patch_shape[1]))
                            y_args.append(y_index + self.patch_shape[1]/2.0)

                            x_args.append(channel_slice)

                            patch_X[i] = X[tuple(x_args)]
                            patch_y[i] = y[tuple(y_args)]

                            x_args = [image_num]
                            y_args = [image_num]
                            i += 1

                            if i == self.num_patches:
                                return
            iteration_count += 1


class MakeC01B(preprocessing.Preprocessor):

    def __init__(self, data_labels=("train_patches", "test_patches", "valid_patches"), y_labels=("train_patch_labels", "test_patch_labels", "valid_patch_labels")):

        self.data_labels = data_labels
        self.y_labels = y_labels

    def apply(self, dataset, can_fit=False):
        print self
        #check if we have already extracted the raw data
        if "c01b_" + self.data_labels[0] in dataset.keys():
            print "skipping extract_raw_data, this has already been run"
            return

        for index in range(len(self.data_labels)):

            data_label = self.data_labels[index]
            y_label = self.y_labels[index]

            print data_label

            num_images = dataset[data_label].shape[0]
            x_dim = dataset[data_label].shape[1]
            y_dim = dataset[data_label].shape[2]
            num_channels = dataset[data_label].shape[3]

            dataset.create_dataset("c01b_" + data_label, (num_channels, x_dim, y_dim, num_images), chunks=(num_channels, x_dim, y_dim, 4))
            dataset["c01b_" + y_label] = dataset[y_label]

            for i in range(num_images):
                if i % (num_images/10) == 0:
                    print "converting to co1b: " + str(i) + "/" + str(num_images)

                b01c_data = np.copy(dataset[data_label][i])
                c01b_data = np.rollaxis(b01c_data, 2)

                dataset["c01b_" + data_label][:, :, :, i] = c01b_data
