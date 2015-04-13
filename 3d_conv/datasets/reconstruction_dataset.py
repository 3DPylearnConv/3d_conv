
import numpy as np
import os
import collections

import binvox_rw


class ReconstructionDataset():

    def __init__(self,
                 models_dir="/srv/3d_conv_data/model_reconstruction/models/",
                 pc_dir="/srv/3d_conv_data/model_reconstruction/pointclouds/",
                 model_name="cordless_drill",
                 patch_size=100):

        self.models_dir = models_dir
        self.pc_dir = pc_dir
        self.model_name = model_name
        self.patch_size = patch_size

        self.model_fullfilename = models_dir + model_name + ".binvox"

        filenames = [d for d in os.listdir(pc_dir + model_name) if not os.path.isdir(os.path.join(pc_dir + model_name, d))]

        self.pointclouds = []
        for file_name in filenames:
                if "_pc.npy" in file_name:

                    pointcloud_file = pc_dir + model_name + "/" + file_name
                    pose_file = pc_dir + model_name + "/" + file_name.replace("pc", "pose")

                    self.pointclouds.append((pointcloud_file, pose_file))

    def get_num_examples(self):
        return len(self.pointclouds)

    def iterator(self,
                 batch_size=None,
                 num_batches=None):

            return ReconstructionIterator(self,
                                          batch_size=batch_size,
                                          num_batches=num_batches)


class ReconstructionIterator(collections.Iterator):

    def __init__(self,
                 dataset,
                 batch_size,
                 num_batches,
                 iterator_post_processors=[]):

        self.dataset = dataset

        self.batch_size = batch_size
        self.num_batches = num_batches

        self.iterator_post_processors = iterator_post_processors

    def __iter__(self):
        return self

    def next(self):

        batch_indices = np.random.random_integers(0, self.dataset.get_num_examples()-1, self.batch_size)

        patch_size = self.dataset.patch_size

        batch_x = np.zeros((self.batch_size, patch_size, patch_size, patch_size, 1))
        batch_y = np.zeros((self.batch_size, patch_size, patch_size, patch_size, 1))

        for i in range(len(batch_indices)):
            index = batch_indices[i]

            model_filepath = self.dataset.model_fullfilename
            pc = np.load(self.dataset.pointclouds[index][0])  # Point cloud. Shape is (number of points, 4). R,G,B,Color
            model_pose = np.load(self.dataset.pointclouds[index][1])  # 4x4 homogeneous transform matrix

            with open(model_filepath, 'rb') as f:
                model = binvox_rw.read_as_3d_array(f)

            batch_y[i, :, :, :, 0] = model.data[:, :, :]

        #make batch B2C01 rather than B012C
        batch_x = batch_x.transpose(0, 3, 4, 1, 2)
        batch_y = batch_y.transpose(0, 3, 4, 1, 2)

        #apply post processors to the patches
        for post_processor in self.iterator_post_processors:
            batch_x, batch_y = post_processor.apply(batch_x, batch_y)

        batch_x = np.array(batch_x, dtype=np.float32)
        batch_y = np.array(batch_y, dtype=np.float32)

        for i in range(len(batch_indices)):
            model_pose = np.load(self.dataset.pointclouds[index][1])  # 4x4 homogeneous transform matrix
            batch_y[i, :, 0, :, :] = rotate_3d(batch_y[[i], :, [0], :, :], model_pose)

        return batch_x, batch_y

    def batch_size(self):
        return self.batch_size

    def num_batches(self):
        return self.num_batches

    def num_examples(self):
        return self.dataset.get_num_examples()

