
import argparse
import sys

import numpy as np
from scipy.misc import imresize

from pylearn2.gui.patch_viewer import make_viewer

#to show, may need to run:
#export PYLEARN2_VIEWER_COMMAND=eog

# weights of shape (out channel, time, in channel, row, column)
def gen_weight_patches(npy_filepath, save_filename):

    weights = np.load(npy_filepath)
    weights = weights[0].get_value()

    s0, s1, s2, s3, s4 = weights.shape
    weights = weights.reshape(s0*s1*s2, s3, s4, 1)

    #not sure this is needed
    weights = weights - weights.min()
    weights = weights / weights.max() * 255

    #this works, but blends the weights
    #weights = scipy.ndimage.zoom(weights, [1, 15, 15, 1], order=3, mode='nearest')

    out = np.zeros((s0*s1*s2, s3*10, s4*10, 1))
    for i in range(s0*s1*s2):
        weight = weights[i, :, :, 0]
        out[i, :, :, 0] = imresize(weight, (s3*10, s4*10), interp='nearest')

    #print weights.shape
    viewer = make_viewer(out)

    if save_filename:
        viewer.save(save_filename)
    else:
        viewer.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('weight_file', type=str, help='weight file to visualize')
    parser.add_argument('--save', type=str, help='save weights file rather than show it. ex: --save out.png', default=None)
    args = parser.parse_args(sys.argv[1:])

    gen_weight_patches(args.weight_file, args.save)