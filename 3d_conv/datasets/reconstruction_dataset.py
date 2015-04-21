
import numpy as np
import os
import collections

import binvox_rw
import visualization.visualize as viz
import tf_conversions
import PyKDL

import math

class ReconstructionDataset():

    def __init__(self,
                 models_dir="/srv/3d_conv_data/model_reconstruction_1000/models/",
                 pc_dir="/srv/3d_conv_data/model_reconstruction_1000/pointclouds/",
                 # models_dir="/srv/3d_conv_data/model_reconstruction_no_rot/models/",
                 # pc_dir="/srv/3d_conv_data/model_reconstruction_no_rot/pointclouds/",
                 # models_dir="/srv/3d_conv_data/model_reconstruction/models/",
                 # pc_dir="/srv/3d_conv_data/model_reconstruction/pointclouds/",
                 model_name="cordless_drill",
                 patch_size=72):

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

def create_voxel_grid_around_point(points, patch_center, voxel_resolution=0.001, num_voxels_per_dim=72):

    voxel_grid = np.zeros((num_voxels_per_dim,
                           num_voxels_per_dim,
                           num_voxels_per_dim,
                           1))

    #could be improved significantly either numpy magic or multi-threaded
    for point in points:

        #get x,y,z indice for the grid
        voxel_index_x, voxel_index_y, voxel_index_z = np.floor((point - patch_center + num_voxels_per_dim/2*voxel_resolution) / voxel_resolution)

        #print voxel_index_x, voxel_index_y, voxel_index_z
        if 0 < voxel_index_x < num_voxels_per_dim :
            if 0 < voxel_index_y < num_voxels_per_dim :
                if 0 < voxel_index_z < num_voxels_per_dim:
                    #mark voxel at this x,y,z indice as occupied.
                    voxel_grid[voxel_index_x, voxel_index_y, voxel_index_z, 0] = 1

    return voxel_grid

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

    def map_pointclouds_to_world(self, pc, non_zero_arr, model_pose):
        #this works, to reorient pointcloud
        #apply the model_pose transform, this is the rotation
        #that was applied to the model in gazebo
        #non_zero_arr1 = np.dot(model_pose, non_zero_arr)
        non_zero_arr1 = non_zero_arr

        #from camera to world
        #the -1 is the fact that the model is 1 meter away from the camera
        dist_to_camera = -1
        trans_frame = PyKDL.Frame(PyKDL.Rotation.RPY(0, 0, 0), PyKDL.Vector(0, 0, dist_to_camera))
        trans_matrix = tf_conversions.posemath.toMatrix(trans_frame)

        #go from camera coords to world coords
        rot_frame = PyKDL.Frame(PyKDL.Rotation.RPY(-math.pi/2, 0, -math.pi/2), PyKDL.Vector(0, 0, 0))
        rot_matrix = tf_conversions.posemath.toMatrix(rot_frame)

        pc2 = np.ones((pc.shape[0], 4))
        pc2[:, 0:3] = pc

        #put point cloud in world frame at origin of world
        pc2_out = np.dot(trans_matrix, pc2.T)
        pc2_out = np.dot(rot_matrix, pc2_out)

        #rotate point cloud by same rotation that model went through
        pc2_out = np.dot(model_pose.T, pc2_out)
        return pc2_out, non_zero_arr1

    def map_pointclouds_to_camera_frame(self, pc, non_zero_arr, model_pose):
        #apply the model_pose transform, this is the rotation
        #that was applied to the model in gazebo
        #non_zero_arr1 = np.dot(model_pose, non_zero_arr)
        non_zero_arr1 = non_zero_arr

        #from camera to world
        #the -1 is the fact that the model is 1 meter away from the camera
        dist_to_camera = -2
        trans_frame = PyKDL.Frame(PyKDL.Rotation.RPY(0, 0, 0), PyKDL.Vector(0, 0, dist_to_camera))
        trans_matrix = tf_conversions.posemath.toMatrix(trans_frame)

        #go from camera coords to world coords
        rot_frame = PyKDL.Frame(PyKDL.Rotation.RPY(-math.pi/2, 0, -math.pi/2), PyKDL.Vector(0, 0, 0))
        rot_matrix = tf_conversions.posemath.toMatrix(rot_frame)

        non_zero_arr1 = np.dot(model_pose, non_zero_arr1)
        non_zero_arr1 = np.dot(rot_matrix.T, non_zero_arr1)
        non_zero_arr1 = np.dot(trans_matrix.T, non_zero_arr1)

        pc2_out = np.ones((pc.shape[0], 4))
        pc2_out[:, 0:3] = pc
        trans_frame = PyKDL.Frame(PyKDL.Rotation.RPY(0, 0, 0), PyKDL.Vector(0, 0, -1))
        trans_matrix = tf_conversions.posemath.toMatrix(trans_frame)
        pc2_out = np.dot(trans_matrix, pc2_out.T)

        return pc2_out, non_zero_arr1

    def next(self):

        batch_indices = np.random.random_integers(0, self.dataset.get_num_examples()-1, self.batch_size)

        patch_size = self.dataset.patch_size

        batch_x = np.zeros((self.batch_size, patch_size, patch_size, patch_size, 1))
        batch_y = np.zeros((self.batch_size, patch_size, patch_size, patch_size, 1))

        for i in range(len(batch_indices)):
            index = batch_indices[i]

            model_filepath = self.dataset.model_fullfilename
            pc = np.load(self.dataset.pointclouds[index][0])  # Point cloud. Shape is (number of points, 4). R,G,B,Color
            #remove 32 bit color channel
            pc = pc[:, 0:3]
            model_pose = np.load(self.dataset.pointclouds[index][1])  # 4x4 homogeneous transform matrix

            with open(model_filepath, 'rb') as f:
                model = binvox_rw.read_as_3d_array(f)

            points = model.data
            scale = model.scale
            translate = model.translate
            dims = model.dims

            non_zero_points = points.nonzero()

            #get numpy array of nonzero points
            num_points = len(non_zero_points[0])
            non_zero_arr = np.zeros((4, num_points))

            for j in range(num_points):
                non_zero_arr[0, j] = non_zero_points[0][j]
                non_zero_arr[1, j] = non_zero_points[1][j]
                non_zero_arr[2, j] = non_zero_points[2][j]
                #added so that we can dot with 4x4 rotation matrix
                non_zero_arr[3, j] = 1.0


            translate_arr = np.array(translate).reshape(3, 1)
            non_zero_arr[0:3, :] = non_zero_arr[0:3, :] + translate_arr

            non_zero_arr[0:3, :] = non_zero_arr[0:3, :] / (scale * 4)

            #this is needed, to recenter binvox model at origin for some reason
            #the translate array does not seem to fully compensate.
            non_zero_arr[2, :] -= .09

            #this is an easier task, the y value is always the same. i.e the model standing
            #up at the origin.
            #pc2_out, non_zero_arr1 = self.map_pointclouds_to_world(pc, non_zero_arr, model_pose)
            pc2_out, non_zero_arr1 = self.map_pointclouds_to_camera_frame(pc, non_zero_arr, model_pose)

            min_x = pc2_out[0, :].min()
            min_y = pc2_out[1, :].min()
            min_z = pc2_out[2, :].min()

            max_x = pc2_out[0, :].max()
            max_y = pc2_out[1, :].max()
            max_z = pc2_out[2, :].max()

            center = (min_x + (max_x-min_x)/2.0, min_y + (max_y-min_y)/2.0, min_z + (max_z-min_z)/2.0)

            # import IPython
            # IPython.embed()
            #now non_zero_arr and pc points are in the same frame of reference.
            #since the images were captured with the model at the origin
            #we can just compute an occupancy grid centered around the origin.
            x = create_voxel_grid_around_point(pc2_out[0:3, :].T, center, voxel_resolution=.02, num_voxels_per_dim=patch_size)
            y = create_voxel_grid_around_point(non_zero_arr1.T[:, 0:3], center, voxel_resolution=.02, num_voxels_per_dim=patch_size)

            # viz.visualize_3d(x)
            # viz.visualize_3d(y)
            # viz.visualize_pointcloud(pc2_out[0:3, :].T)
            # viz.visualize_pointclouds(pc2_out.T, non_zero_arr1.T[:, 0:3], False, True)
            # import IPython
            # IPython.embed()

            batch_y[i, :, :, :, :] = y
            batch_x[i, :, :, :, :] = x

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

