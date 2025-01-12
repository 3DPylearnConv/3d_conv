

import numpy as np

import binvox_rw
import visualization.visualize as viz
import tf_conversions
import PyKDL
from off_utils.off_handler import OffHandler
import math


def create_voxel_grid_around_point(points, patch_center, voxel_resolution=0.001, num_voxels_per_dim=72):

    voxel_grid = np.zeros((num_voxels_per_dim,
                           num_voxels_per_dim,
                           num_voxels_per_dim,
                           1))

    centered_scaled_points = np.floor((points-patch_center + num_voxels_per_dim/2*voxel_resolution) / voxel_resolution)

    mask = centered_scaled_points.max(axis=1) < num_voxels_per_dim
    centered_scaled_points = centered_scaled_points[mask]

    if centered_scaled_points.shape[0] == 0:
        return voxel_grid

    mask = centered_scaled_points.min(axis=1) > 0
    centered_scaled_points = centered_scaled_points[mask]

    if centered_scaled_points.shape[0] == 0:
        return voxel_grid

    csp_int = centered_scaled_points.astype(int)

    mask = (csp_int[:, 0], csp_int[:, 1], csp_int[:, 2], np.zeros((csp_int.shape[0]), dtype=int))

    voxel_grid[mask] = 1


    return voxel_grid


def map_pointclouds_to_world(pc, non_zero_arr, model_pose):
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


def map_pointclouds_to_camera_frame(pc, non_zero_arr, model_pose):
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

    non_zero_arr1 = np.dot(model_pose, non_zero_arr1)
    non_zero_arr1 = np.dot(rot_matrix.T, non_zero_arr1)
    non_zero_arr1 = np.dot(trans_matrix.T, non_zero_arr1)

    pc2_out = np.ones((pc.shape[0], 4))
    pc2_out[:, 0:3] = pc
    trans_frame = PyKDL.Frame(PyKDL.Rotation.RPY(0, 0, 0), PyKDL.Vector(0, 0, -1))
    trans_matrix = tf_conversions.posemath.toMatrix(trans_frame)
    pc2_out = np.dot(trans_matrix, pc2_out.T)

    return pc2_out, non_zero_arr1


def build_training_example(binvox_file_path, model_pose_filepath, single_view_pointcloud_filepath, patch_size, custom_scale=1, custom_offset=(0, 0, 0)):
    custom_offset = np.array(custom_offset).reshape(3,1)

    pc = np.load(single_view_pointcloud_filepath)
    pc = pc[:, 0:3]
    model_pose = np.load(model_pose_filepath)
    with open(binvox_file_path, 'rb') as f:
        model = binvox_rw.read_as_3d_array(f)

    points = model.data
    binvox_scale = model.scale
    binvox_offset = model.translate
    dims = model.dims

    non_zero_points = points.nonzero()

    #get numpy array of nonzero points
    num_points = len(non_zero_points[0])
    non_zero_arr = np.zeros((4, num_points))

    non_zero_arr[0] = non_zero_points[0]
    non_zero_arr[1] = non_zero_points[1]
    non_zero_arr[2] = non_zero_points[2]
    non_zero_arr[3] = 1.0

    binvox_offset = np.array(binvox_offset).reshape(3, 1)

    #go from binvox to off original mesh
    non_zero_arr[0:3, :] = non_zero_arr[0:3, :] / np.array(dims).reshape(3, 1) * binvox_scale
    non_zero_arr[0:3, :] = non_zero_arr[0:3, :] + binvox_offset

    # oh = OffHandler()
    # oh.read("/home/jvarley/.gazebo/models/D00532/D00532.off")
    # viz.visualize_pointclouds(oh.vertices, non_zero_arr.T[:, 0:3], False, True)

    #go from off to mesh
    non_zero_arr[0:3, :] = non_zero_arr[0:3, :] * custom_scale
    non_zero_arr[0:3, :] = non_zero_arr[0:3, :] - custom_offset


    #this is an easier task, the y value is always the same. i.e the model standing
    #up at the origin.
    #pc2_out, non_zero_arr1 = map_pointclouds_to_world(pc, non_zero_arr, model_pose)
    pc2_out, non_zero_arr1 = map_pointclouds_to_camera_frame(pc, non_zero_arr, model_pose)

    min_x = pc2_out[0, :].min()
    min_y = pc2_out[1, :].min()
    min_z = pc2_out[2, :].min()

    max_x = pc2_out[0, :].max()
    max_y = pc2_out[1, :].max()
    max_z = pc2_out[2, :].max()

    center = (min_x + (max_x-min_x)/2.0, min_y + (max_y-min_y)/2.0, min_z + (max_z-min_z)/2.0)

    viz.visualize_pointclouds(pc2_out.T, non_zero_arr1.T[:, 0:3], False, True)

    # import IPython
    # IPython.embed()
    # assert False

    #now non_zero_arr and pc points are in the same frame of reference.
    #since the images were captured with the model at the origin
    #we can just compute an occupancy grid centered around the origin.
    x = create_voxel_grid_around_point(pc2_out[0:3, :].T, center, voxel_resolution=.015, num_voxels_per_dim=patch_size)
    y = create_voxel_grid_around_point(non_zero_arr1.T[:, 0:3], center, voxel_resolution=.015, num_voxels_per_dim=patch_size)

    return x, y
