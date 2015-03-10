# 3d_conv

## Calendar
April 14 - Final presentation due

# 3d Convolution for Pylearn2
## Project Overview
####contact_and_potential_grasps_small.h5
A small dataset containing 8 grasps.  Each grasp has an rgbd image, and 17 uvd tuples.  The uvd values are the pixel and depth values for the positions of the virtual contacts of the hand within the image.  The uvd values at indices 0, 8,12,16 correspond to the palm, and three fingertips.

####rgbd_hdf5_dataset.py
Wraps around an hdf5 datafile, and provides pairs of patches and labels at training time.

####train_model.py
Call this to train a model.  It allows you to specify a model.yaml file as well as a dataset to train with.  The model is selected out of the model_templates directory and copied into the models directory.

####model_templates directory
Contains .yaml files specifying layer configurations, learning rates, and other parameters needed to train a model

####models directory
When a model is trained, all yaml files from the model_templates directory are copied over here.  The .pkl file for the model is also stored here, along with images of the weights generated as the model trains.

####set_paths.sh
Places this projects code on your PYTHONPATH and also sets HOME_PATH which is used in paths.py to determine where different directories are located.