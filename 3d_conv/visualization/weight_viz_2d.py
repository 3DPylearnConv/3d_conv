
import numpy as np
import scipy.ndimage

from pylearn2.gui.patch_viewer import make_viewer

#to show, may need to run:
#export PYLEARN2_VIEWER_COMMAND=eog

# weights of shape (out channel, time, in channel, row, column)
def gen_weight_patches(weights, save=False):

    #weights = np.random.rand(20, 3, 3, 1)
    s0, s1, s2, s3, s4 = weights.shape
    weights = weights.reshape(s0*s1*s2, s3, s4, 1)
    weights = scipy.ndimage.zoom(weights, [1, 4, 4, 1], order=0)
    viewer = make_viewer(weights)

    if save:
        viewer.save()
    else:
        viewer.show()