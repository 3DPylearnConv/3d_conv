from stl import stl
import numpy as np

import visualization.visualize as viz

from utils.reconstruction_utils import create_voxel_grid_around_point


class GraspitDataset():

    def __init__(self, patch_size=24):

        self.examples = []
        #self.examples.append('/media/Elements/captured_meshes/1425077494_all_bottle_6/model_4.stl')
        #self.examples.append('/media/Elements/captured_meshes/1425148122_shampoo_1/model_6.stl')
        self.examples.append('/media/Elements/captured_meshes/1425081374_drill_1/model_4.stl')

        self.patch_size = patch_size

    def get_num_examples(self):
        return len(self.examples)

    def iterator(self,
                 batch_size=None,
                 num_batches=None):

            return GraspitIterator(self,
                                 batch_size=batch_size,
                                 num_batches=num_batches)


class GraspitIterator():

    def __init__(self, dataset,
                 batch_size,
                 num_batches):


        self.dataset = dataset
        self.batch_size = batch_size
        self.num_batches = num_batches

    def __iter__(self):
        return self

    def next(self):

        batch_indices = np.random.random_integers(0, self.dataset.get_num_examples()-1, self.batch_size)

        patch_size = self.dataset.patch_size

        batch_x = np.zeros((self.batch_size, patch_size, patch_size, patch_size, 1), dtype=np.float32)

        for i in range(len(batch_indices)):
            index = batch_indices[i]
            model_filepath = self.dataset.examples[index]

            mesh = stl.StlMesh(model_filepath)
            pts = mesh.points.reshape(mesh.points.shape[0]*3, 3)

            patch_center = (pts[:, 0].mean(), pts[:, 1].mean(), pts[:, 2].mean())
            voxel_grid = create_voxel_grid_around_point(pts, patch_center, voxel_resolution=8, num_voxels_per_dim=patch_size)

            batch_x[i, :, :, :, :] = voxel_grid

        #make batch C01B rather than B01C
        batch_x = batch_x.transpose(0, 3, 4, 1, 2)

        return batch_x

    def batch_size(self):
        return self.batch_size

    def num_batches(self):
        return self.num_batches

    def num_examples(self):
        return self.dset.get_num_examples()
