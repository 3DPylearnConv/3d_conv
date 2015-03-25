
import numpy as np


class OffHandler():

    def read(self, filepath):

        self.vertices = []
        self.faces = []
        f = open(filepath, 'r')

        f.readline()

        v_count, f_count, e_count = [int(x) for x in f.readline().split()]

        i = 0
        while i < v_count:
            line = f.readline()
            px, py, pz = [float(x) for x in line.split()]

            self.vertices.append((px, py, pz))
            i += 1

        i = 0
        while i < f_count:
            line = f.readline()
            num_indices, vid_0, vid_1, vid_2 = [float(x) for x in line.split()]

            self.faces.append((vid_0, vid_1, vid_2))
            i += 1

    def get_centroid(self):
        vertices_arr = np.array(self.vertices)
        return np.average(vertices_arr[:,0]), np.average(vertices_arr[:,1]), np.average(vertices_arr[:, 2])

    def get_full_vertices(self):
        return self.vertices

    def get_half_model(self):
        centroid = self.get_centroid()

        out_points = []

        for v in self.vertices:
            if v[2] < centroid[2]:
                out_points.append(v)

        return out_points



