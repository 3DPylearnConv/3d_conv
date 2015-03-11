
from rgbd_hdf5_dataset import RGBD_HDF5_Dataset, HDF5_Iterator,  GaussianNoisePostProcessor

import numpy as np


class PointCloud_HDF5_Dataset(RGBD_HDF5_Dataset):

    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 topo=None, targets=None, rng=None, data_specs=None,
                 return_tuple=False):

        return HDF5_PointCloud_Iterator(self,
                             batch_size=batch_size,
                             num_batches=num_batches,
                             mode=mode)


def get_camera_info(hard_coded=True):

    if hard_coded:
        cx = 319.5
        cy = 239.5
        fx = 525.5
        fy = 525.5

        return (cx, cy, fx, fy)

    #if we are using a different camera, then
    #we can listen to the ros camera info topic for that device
    #and get our values here.
    else:

        import image_geometry
        from sensor_msgs.msg import CameraInfo

        cam_info = CameraInfo()

        cam_info.height = 480
        cam_info.width = 640
        cam_info.distortion_model = "plumb_bob"
        cam_info.D = [0.0, 0.0, 0.0, 0.0, 0.0]
        cam_info.K = [525.0, 0.0, 319.5, 0.0, 525.0, 239.5, 0.0, 0.0, 1.0]
        cam_info.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        cam_info.P = [525.0, 0.0, 319.5, 0.0, 0.0, 525.0, 239.5, 0.0, 0.0, 0.0, 1.0, 0.0]
        cam_info.binning_x = 0
        cam_info.binning_y = 0
        cam_info.roi.x_offset = 0
        cam_info.roi.y_offset = 0
        cam_info.roi.height = 0
        cam_info.roi.width = 0
        cam_info.roi.do_rectify = False

        camera_model = image_geometry.PinholeCameraModel()
        camera_model.fromCameraInfo(cam_info)

        return camera_model.cx(), camera_model.cy(), camera_model.fx(), camera_model.fy()


#this function takes an rgbd_image
#and creates a 3d point cloud out of it
#using the parameters of the camera used to capture the
#rgbd image
def create_point_cloud_vectorized(rgbd_image, structured=False):

    cx, cy, fx, fy = get_camera_info()
    im_shape = rgbd_image.shape

    # get the depth
    d = rgbd_image[:, :, 3]

    # replace the invalid data with np.nan
    z = np.where((d > 0) & (d < 255), d, np.nan)

    # get x and y data in a vectorized way
    x = (np.arange(im_shape[0])[:, None] - cx) / fx * z
    y = (np.arange(im_shape[1])[None, :] - cy) / fy * z

    if structured:
        #if we want (480,640, 3) i.e. (x,y,z)
        return np.array((x, y, z)).transpose(1, 2, 0)

    #if we want large array (num_points, 3)
    return np.array((x, y, z)).reshape(3, -1).swapaxes(0, 1)


#this creates a 3d occupancy grid based on an rgbd image.
def create_voxel_grid_around_point(rgbd, patch_center, voxel_resolution=1.0, voxel_grid_dimension=72):

    points = create_point_cloud_vectorized(rgbd, structured=False)

    voxel_grid = np.zeros((voxel_grid_dimension,
                           voxel_grid_dimension,
                           voxel_grid_dimension,
                            1))

    #could be improved significantly either numpy magic or multi-threaded
    for point in points:

        #get x,y,z indice for the grid
        voxel_index_x, voxel_index_y, voxel_index_z = np.floor((point - patch_center + voxel_grid_dimension/2) / voxel_resolution) * voxel_resolution

        #mark voxel at this x,y,z indice as occupied.
        voxel_grid[voxel_index_x, voxel_index_y, voxel_index_z, 0] = 1

    return voxel_grid


class HDF5_PointCloud_Iterator(HDF5_Iterator):

    def next(self):

        batch_indices = self._subset_iterator.next()

        if isinstance(batch_indices, slice):
            batch_indices = np.array(range(batch_indices.start, batch_indices.stop))

        # if we are using a shuffled sequential subset iterator
        # then next_index will be something like:
        # array([13713, 14644, 30532, 32127, 35746, 44163, 48490, 49363, 52141, 52216])
        # hdf5 can only support this sort of indexing if the array elements are
        # in increasing order
        batch_size = 0
        if isinstance(batch_indices, np.ndarray):
            batch_indices.sort()
            batch_size = len(batch_indices)

        num_uvd_per_rgbd = self.dataset.h5py_dataset['uvd'].shape[1]
        num_grasp_types = self.dataset.h5py_dataset['num_grasp_type'][0]

        finger_indices = batch_indices % num_uvd_per_rgbd
        batch_indices = np.floor(batch_indices / num_uvd_per_rgbd)

        patch_size = self.dataset.patch_size

        batch_x = np.zeros((batch_size, patch_size, patch_size, patch_size, 1))
        batch_y = np.zeros((batch_size, num_uvd_per_rgbd * num_grasp_types))

        #go through and append patches to batch_x, batch_y
        for i in range(len(finger_indices)):
            batch_index = batch_indices[i]
            finger_index = finger_indices[i]
            u, v, d = self.dataset.h5py_dataset['uvd'][batch_index, finger_index, :]
            rgbd = self.dataset.topo_view[batch_index, :, :, :]

            structured_points = create_point_cloud_vectorized(rgbd, True)
            patch_center_x, patch_center_y, patch_center_z = structured_points[u, v]

            patch = create_voxel_grid_around_point(rgbd, (patch_center_x, patch_center_y, patch_center_z),voxel_grid_dimension=patch_size)

            grasp_type = self.dataset.y[batch_index, 0]
            grasp_energy = self.dataset.h5py_dataset['energy'][batch_index]

            patch_label = num_uvd_per_rgbd * grasp_type + finger_index

            batch_x[i, :, :, :] = patch
            batch_y[i, patch_label] = grasp_energy

        #make batch C012B rather than B012C
        batch_x = batch_x.transpose(4, 1, 2, 3, 0)

        #apply post processors to the patches
        for post_processor in self.iterator_post_processors:
            batch_x, batch_y = post_processor.apply(batch_x, batch_y)

        batch_x = np.array(batch_x, dtype=np.float32)
        batch_y = np.array(batch_y, dtype=np.float32)

        return batch_x, batch_y

if __name__ == "__main__":
    import IPython
    IPython.embed()
