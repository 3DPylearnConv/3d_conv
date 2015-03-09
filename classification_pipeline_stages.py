
import theano
import pylearn2
import cPickle
import copy
import numpy as np
import h5py
import time
import scipy.signal
import scipy.misc
from scipy.signal import argrelextrema
from theano.tensor.nnet import conv

import matplotlib.pyplot as plt
from subtractive_divisive_lcn import *

import cPickle
import pylearn2.models.mlp


class ClassificationStage():

    def __init__(self, in_key, out_key):
        self.in_key = in_key
        self.out_key = out_key

    def dataset_inited(self, dataset):
        return self.out_key in dataset.keys()

    def init_dataset(self, dataset):
        out = self._run(dataset, 0)

        shape = (900, out.shape[0], out.shape[1], out.shape[2])
        chunk_size = (10, out.shape[0], out.shape[1], out.shape[2])

        dataset.create_dataset(self.out_key, shape, chunks=chunk_size)

    def run(self, dataset, index):
        out = self._run(dataset, index)
        dataset[self.out_key][index] = out

    def _run(self, dataset, index):
        print "Base class _run should not be called."
        raise NotImplementedError


class CopyInRaw(ClassificationStage):

    def __init__(self, raw_rgbd_dataset_filepath, in_key='images', out_key='rgbd_data'):
        self.raw_rgbd_dataset = h5py.File(raw_rgbd_dataset_filepath)
        self.in_key = in_key
        self.out_key = out_key

    def init_dataset(self, dataset):
        shape = self.raw_rgbd_dataset[self.in_key].shape
        if shape[0] < 10:
            num_samples_per_chunk = shape[0]
        else:
            num_samples_per_chunk = 10
        chunk_size = (num_samples_per_chunk, shape[1], shape[2], shape[3])
        dataset.create_dataset(self.out_key, shape, chunks=chunk_size)

    def run(self, dataset, index):
        dataset[self.out_key][index] = self.raw_rgbd_dataset[self.in_key][index]


class LecunSubtractiveDivisiveLCN(ClassificationStage):

    def __init__(self, kernel_shape=9, in_key='rgbd_data', out_key='rgbd_data_normalized'):
        self.kernel_shape = kernel_shape
        self.in_key = in_key
        self.out_key = out_key
        self.sub_div_lcn = None

    def _run(self, dataset, index):
        img = dataset[self.in_key][index]
        num_channels = img.shape[2]

        img_out = np.zeros_like(img)
        img_in = np.zeros((1, img.shape[0], img.shape[1]), dtype=np.float32)

        if not self.sub_div_lcn:
            self.sub_div_lcn = subtractive_divisive_lcn(img_in, img_shape=img.shape[0:2], kernel_shape=self.kernel_shape)

        for i in range(num_channels):
            img_in[0] = img[:, :, i]
            img_out[:, :, i] = self.sub_div_lcn(img_in.reshape((img_in.shape[0], img_in.shape[1], img_in.shape[2], 1)))

        return img_out


class FeatureExtraction(ClassificationStage):

    def __init__(self, model_filepath,
                 in_key='rgbd_data_normalized',
                 out_key='extracted_features',
                 use_float_64=False):

        ClassificationStage.__init__(self, in_key, out_key)

        self.model_filepath = model_filepath

        if use_float_64:
            self.float_type_str = 'float64'
            self.float_dtype = np.float64

        else:
            self.float_type_str = 'float32'
            self.float_dtype = np.float32

        # this is defined in init_dataset
        # we need to know the size of the input
        # in order to init this
        self._feature_extractor = None
        self.shape = None
        self.num_channels = None

    def init_dataset(self, dataset):

        f = open(self.model_filepath)

        cnn_model = cPickle.load(f)

        img_in = dataset[self.in_key][0]
        self.shape = img_in.shape[0:2]
        self.num_channels = img_in.shape[-1]

        new_space = pylearn2.space.Conv2DSpace(self.shape, num_channels=self.num_channels, axes=('c', 0, 1, 'b'), dtype=self.float_type_str)

        start_classifier_index = 0
        for i in range(len(cnn_model.layers)):
            if not isinstance(cnn_model.layers[i], pylearn2.models.mlp.ConvRectifiedLinear):
                start_classifier_index = i
                break

        cnn_model.layers = cnn_model.layers[0:start_classifier_index]

        weights = []
        biases = []

        for layer in cnn_model.layers:
            weights.append(np.copy(layer.get_weights_topo()))
            biases.append(np.copy(layer.get_biases()))



        cnn_model.set_batch_size(1)
        cnn_model.set_input_space(new_space)

        for i in range(len(cnn_model.layers)):
            weights_rolled = np.rollaxis(weights[i], 3, 1)
            cnn_model.layers[i].set_weights(weights_rolled)
            cnn_model.layers[i].set_biases(biases[i])

        X = cnn_model.get_input_space().make_theano_batch()
        Y = cnn_model.fprop(X)

        self._feature_extractor = theano.function([X], Y)

        ClassificationStage.init_dataset(self, dataset)

    def _run(self, dataset, index):
        img_in = dataset[self.in_key][index]

        img = np.zeros((self.num_channels, self.shape[0], self.shape[1], 1), dtype=self.float_dtype)

        img[:, :, :, 0] = np.rollaxis(img_in, 2, 0)

        out_raw = self._feature_extractor(img)

        out_rolled = np.rollaxis(out_raw, 1, 4)
        out_window = out_rolled[0, :, :, :]

        return out_window


class Classification(ClassificationStage):

    def __init__(self, model_filepath, in_key='extracted_features', out_key='heatmaps' ):

        self.in_key = in_key
        self.out_key = out_key

        f = open(model_filepath)

        cnn_model = cPickle.load(f)

        start_classifier_index = 0
        for i in range(len(cnn_model.layers)):
            if not isinstance(cnn_model.layers[i], pylearn2.models.mlp.ConvRectifiedLinear):
                start_classifier_index = i
                break

        layers = cnn_model.layers[start_classifier_index:]

        self.Ws = []
        self.bs = []

        self.Ws.append(layers[0].get_weights_topo())
        self.bs.append(layers[0].get_biases())
        for i in range(len(layers)):
            if i != 0:
                layer = layers[i]
                print type(layer)
                self.Ws.append(layer.get_weights())
                self.bs.append(layer.get_biases())

    def _run(self, dataset, index):

        X = dataset[self.in_key][index]

        W0 = self.Ws[0]
        out = np.dot(X, W0)[:, :, :, 0, 0] + self.bs[0]

        for i in range(len(self.Ws)):
            if i != 0:
                out = np.dot(out, self.Ws[i]) + self.bs[i]

        return out


class HeatmapNormalization(ClassificationStage):

    def __init__(self, in_key='heatmaps', out_key='normalized_heatmaps'):
        self.in_key = in_key
        self.out_key = out_key
        self.max = 255.0

    def _run(self, dataset, index):
        heatmaps = dataset[self.in_key][index]
        normalize_heatmaps = self.max-(heatmaps-heatmaps.min())/(heatmaps.max()-heatmaps.min())*self.max
        return normalize_heatmaps





class Rescale(ClassificationStage):

    def __init__(self, in_key, out_key, model_filepath):
        self.in_key = in_key
        self.out_key = out_key
        f = open(model_filepath)
        model = cPickle.load(f)

        self.pool_strides = []
        self.pool_shapes = []
        self.kernel_strides = []
        self.kernel_shapes = []

        for layer in model.layers:
            if isinstance(layer, pylearn2.models.mlp.ConvRectifiedLinear):
                self.pool_strides.append(layer.pool_stride)
                self.pool_shapes.append(layer.pool_shape)
                self.kernel_shapes.append(layer.kernel_shape)
                self.kernel_strides.append(layer.kernel_stride)

        self.pool_strides.reverse()
        self.pool_shapes.reverse()
        self.kernel_strides.reverse()
        self.kernel_shapes.reverse()

    def init_dataset(self, dataset):

        heatmaps = self._run(dataset, 0)
        num_examples = dataset[self.in_key].shape[0]

        shape = (num_examples, heatmaps.shape[0], heatmaps.shape[1], heatmaps.shape[2])

        dataset.create_dataset(self.out_key,
                               shape,
                               chunks=(10, heatmaps.shape[0], heatmaps.shape[1], heatmaps.shape[2]))

    def _run(self, dataset, index):

        def expand(heatmap, rescale_factor):
            shape = heatmap.shape
            new_shape = (shape[0] * rescale_factor[0], shape[1] * rescale_factor[1])
            return scipy.misc.imresize(heatmap, new_shape)


        heatmap = dataset[self.in_key][index]
        num_conv_layers = len(self.pool_shapes)
        num_channels = heatmap.shape[-1]

        # Intial scale factor.
        scale_factor = [1, 1]
        for layer_index in range(num_conv_layers):
            scale_factor[0] = scale_factor[0] * self.kernel_strides[layer_index][0] * self.pool_strides[layer_index][0]
            scale_factor[1] = scale_factor[1] * self.kernel_strides[layer_index][1] * self.pool_strides[layer_index][1]

        x_dim = heatmap.shape[0] * scale_factor[0]
        y_dim = heatmap.shape[1] * scale_factor[1]

        heatmap_out = np.zeros((x_dim, y_dim, num_channels))

        for channel_index in range(num_channels):
            heatmap_out[:, :, channel_index] = expand(heatmap[:, :, channel_index], scale_factor)

        """
                heatmap_out = np.zeros(heatmap.shape * scale_factor)
                #heatmap at this point is np.array of dim (20,20,14)
                for layer_index in range(num_conv_layers):
                    for channel_index in range(num_channels):
                        expanded_heatmap = np.zeros()
                        heatmap[:,:,channel_index] = expand(heatmap[:,:,channel_index], self.pool_strides[layer_index])
                        heatmap = expand(heatmap, self.kernel_strides[layer_index])

                #we want heatmap at this point to be (
        """
        return heatmap_out


class ConvolveHorizontalGraspPriors(ClassificationStage):

    def __init__(self, priors_filepath):
        self.priors = h5py.File(priors_filepath)
        l_g_p = self.priors['l_gripper_given_palm_blur_norm'][100:200, 100:200]
        l_g_r = self.priors['l_gripper_given_r_gripper_blur_norm'][100:200, 100:200]

        p_g_l = self.priors['palm_given_l_gripper_blur_norm'][100:200, 100:200]
        p_g_r = self.priors['palm_given_r_gripper_blur_norm'][100:200, 100:200]

        r_g_l = self.priors['r_gripper_given_l_gripper_blur_norm'][100:200, 100:200]
        r_g_p = self.priors['r_gripper_given_palm_blur_norm'][100:200, 100:200]

        input = theano.tensor.tensor4(name='input')
        w_shape = (6, 1, 100, 100)
        w = np.zeros(w_shape)
        w[0, 0] = l_g_p
        w[1, 0] = l_g_r

        w[2, 0] = p_g_l
        w[3, 0] = p_g_r

        w[4, 0] = r_g_l
        w[5, 0] = r_g_p

        W = theano.shared(np.asarray(w, dtype=input.dtype), name='W')
        self.conv_out = conv.conv2d(input, W)
        self.f = theano.function([input], self.conv_out)

    def dataset_inited(self,dataset):
        return 'l_convolved_heatmaps' in dataset.keys()

    def init_dataset(self, dataset):

        l_gripper_conv, palm_conv, r_gripper_conv, l_gripper_out, palm_out, r_gripper_out = self._run(dataset, 0)

        shape = (900, l_gripper_conv.shape[1], l_gripper_conv.shape[2], l_gripper_conv.shape[3])
        chunk_size = (10, l_gripper_conv.shape[1], l_gripper_conv.shape[2], l_gripper_conv.shape[3])

        dataset.create_dataset("l_convolved_heatmaps", shape, chunks=chunk_size)
        dataset.create_dataset("r_convolved_heatmaps", shape, chunks=chunk_size)
        dataset.create_dataset("p_convolved_heatmaps", shape, chunks=chunk_size)

        shape = (900, palm_out.shape[0], palm_out.shape[1], 3)
        chunk_size = (10, palm_out.shape[0], palm_out.shape[1], 3)
        dataset.create_dataset("convolved_heatmaps", shape, chunks=chunk_size)

    def _run(self, dataset, index):
        heatmaps = dataset['rescaled_heatmaps'][index]
        # img_in_shape = (416, 576)
        # l_gripper_obs = scipy.misc.imresize(heatmaps[:, :, 0], img_in_shape)
        # palm_obs = scipy.misc.imresize(heatmaps[:, :, 1], img_in_shape)
        # r_gripper_obs = scipy.misc.imresize(heatmaps[:, :, 2], img_in_shape)

        l_gripper_obs = heatmaps[:, :, 0]
        palm_obs = heatmaps[:, :, 1]
        r_gripper_obs = heatmaps[:, :, 2]
        img_in_shape = heatmaps.shape[:-1]


        img_in = np.zeros((1, 1, img_in_shape[0], img_in_shape[1]), dtype=np.float32)

        img_in[:, :] = l_gripper_obs
        l_gripper_conv = self.f(img_in)

        img_in[:, :] = r_gripper_obs
        r_gripper_conv = self.f(img_in)

        img_in[:, :] = palm_obs
        palm_conv = self.f(img_in)

        out_shape = palm_conv[0, 0].shape
        x_border = (img_in_shape[0]-out_shape[0])/2
        y_border = (img_in_shape[1]-out_shape[1])/2

        l_gripper_obs2 = l_gripper_obs[x_border:-x_border - 1, y_border:-y_border - 1]
        palm_obs2 = palm_obs[x_border:-x_border - 1, y_border:-y_border - 1]
        r_gripper_obs2 = r_gripper_obs[x_border:-x_border - 1, y_border:-y_border - 1]

        l_gripper_out = l_gripper_obs2 * palm_conv[0, 0] * r_gripper_conv[0, 1]
        palm_out = palm_obs2 * l_gripper_conv[0, 2] * r_gripper_conv[0, 3]
        r_gripper_out = r_gripper_obs2 * l_gripper_conv[0, 4] * palm_conv[0, 5]

        return l_gripper_conv, palm_conv, r_gripper_conv, l_gripper_out, palm_out, r_gripper_out

    def run(self, dataset, index):

        l_gripper_conv, palm_conv, r_gripper_conv, l_gripper_out, palm_out, r_gripper_out = self._run(dataset, index)

        dataset["l_convolved_heatmaps"][index] = l_gripper_conv
        dataset["r_convolved_heatmaps"][index] = r_gripper_conv
        dataset["p_convolved_heatmaps"][index] = palm_conv

        dataset['convolved_heatmaps'][index, :, :, 0] = l_gripper_out
        dataset['convolved_heatmaps'][index, :, :, 1] = palm_out
        dataset['convolved_heatmaps'][index, :, :, 2] = r_gripper_out


class ConvolveGraspPriors(ClassificationStage):

    def __init__(self, priors_filepath):
        self.out_key = 'convolved_heatmaps'
        dset = h5py.File(priors_filepath)
        priors = dset['priors']
        self.num_grasp_types = priors.shape[0]
        self.num_priors_per_grasp = priors.shape[1]


        w_shape = (self.num_priors_per_grasp, 1, 100, 100)
        self.fs = []

        for i in range(self.num_grasp_types):
            input = theano.tensor.tensor4(name='input' + str(i))

            w = np.zeros(w_shape)
            w[:, 0] = priors[i, :, 100:200, 100:200]

            W = theano.shared(np.asarray(w, dtype=input.dtype), name='W' + str(i))
            conv_out = conv.conv2d(input, W)
            self.fs.append(theano.function([input], conv_out))

    def dataset_inited(self,dataset):
        return 'convolved_heatmaps' in dataset.keys()

    def init_dataset(self, dataset):

        convolved_heatmaps = self._run(dataset, 0)

        shape = (900, convolved_heatmaps.shape[0], convolved_heatmaps.shape[1], convolved_heatmaps.shape[2], convolved_heatmaps.shape[3])
        chunk_size = (10, convolved_heatmaps.shape[0], convolved_heatmaps.shape[1], convolved_heatmaps.shape[2], convolved_heatmaps.shape[3])

        dataset.create_dataset("convolved_heatmaps", shape)


    def _run(self, dataset, index):
        heatmaps = dataset['rescaled_heatmaps'][index]

        img_in_shape = heatmaps.shape[:-1]
        num_heatmaps = heatmaps.shape[-1]

        img_in = np.zeros((1, 1, img_in_shape[0], img_in_shape[1]), dtype=np.float32)

        out = None

        #we have a heatmap for each grasp type for each finger
        #we have priors for each grasp type for each finger for each finger
        for i in range(num_heatmaps):
            conv_function = self.fs[i / self.num_priors_per_grasp]
            img_in[:, :] = heatmaps[:, :, i]
            convolved_heatmaps = conv_function(img_in)
            if out is None:
                out = np.zeros((num_heatmaps, self.num_priors_per_grasp, convolved_heatmaps.shape[-2], convolved_heatmaps.shape[-1]))

            out[i] = convolved_heatmaps[0]

        return out

    # def run(self, dataset, index):
    #     convolved_priors = self._run(dataset,index)
    #     dataset['convolved_priors'][index] = convolved_priors


class ConvolveBarrettPriors(ClassificationStage):

    def __init__(self, priors_filepath):
        self.out_key = 'convolved_heatmaps'
        dset = h5py.File(priors_filepath)
        priors = dset['priors']
        self.num_grasp_types = priors.shape[0]
        self.num_priors_per_grasp = priors.shape[1]

        prior_xdim = priors.shape[-2]
        prior_ydim = priors.shape[-1]

        w_shape = (self.num_priors_per_grasp**2, 1, 100, 100)
        self.fs = []

        for i in range(self.num_grasp_types):

            input = theano.tensor.tensor4(name='input' + str(i))
            w = np.zeros(w_shape)

            for j in range(self.num_priors_per_grasp):
                for k in range(self.num_priors_per_grasp):

                    w[j*self.num_priors_per_grasp + k, 0] = priors[i, j, k, int(prior_xdim/2.0-50):int(prior_xdim/2.0+50), int(prior_ydim/2.0-50):int(prior_ydim/2.0+50)]

            W = theano.shared(np.asarray(w, dtype=input.dtype), name='W' + str(i))
            conv_out = conv.conv2d(input, W)
            self.fs.append(theano.function([input], conv_out))

    def dataset_inited(self,dataset):
        return 'convolved_heatmaps' in dataset.keys()

    def init_dataset(self, dataset):

        convolved_heatmaps = self._run(dataset, 0)
        shape = (900, convolved_heatmaps.shape[0], convolved_heatmaps.shape[1], convolved_heatmaps.shape[2], convolved_heatmaps.shape[3],convolved_heatmaps.shape[4])
        chunk_size = (10, convolved_heatmaps.shape[0], convolved_heatmaps.shape[1], convolved_heatmaps.shape[2], convolved_heatmaps.shape[3],convolved_heatmaps.shape[4])

        dataset.create_dataset("convolved_heatmaps", shape)


    def _run(self, dataset, index):
        heatmaps = dataset['rescaled_heatmaps'][index]

        img_in_shape = heatmaps.shape[:-1]
        num_heatmaps = heatmaps.shape[-1]

        img_in = np.zeros((1, 1, img_in_shape[0], img_in_shape[1]), dtype=np.float32)

        out = None

        #we have a heatmap for each grasp type for each finger
        #we have priors for each grasp type for each finger for each finger
        for i in range(num_heatmaps):
            conv_function = self.fs[i / self.num_priors_per_grasp]
            img_in[:, :] = heatmaps[:, :, i]
            convolved_heatmaps = conv_function(img_in)

            if out is None:
                #12 heatmaps, #4 convolutions for each heatmap #heatmapxdim, #heatmapydim
                out = np.zeros((num_heatmaps, self.num_priors_per_grasp, self.num_priors_per_grasp, convolved_heatmaps.shape[-2], convolved_heatmaps.shape[-1]))

            for j in range(self.num_priors_per_grasp):
                for k in range(self.num_priors_per_grasp):
                    index = j * self.num_priors_per_grasp + k
                    out[i, j, k] = convolved_heatmaps[0, index]

        return out



class BarrettMultiplyPriors(ClassificationStage):

    def dataset_inited(self, dataset):
        return 'independent_x_priors' in dataset.keys()

    def init_dataset(self, dataset):
        independent_x_priors = self._run(dataset, 0 )

        shape = (900, independent_x_priors.shape[0], independent_x_priors.shape[1], independent_x_priors.shape[2])
        dataset.create_dataset('independent_x_priors', shape)

    def _run(self, dataset, index):
        priors = dataset['convolved_heatmaps'][index]
        independent = dataset['rescaled_heatmaps'][index]

        num_heatmaps = priors.shape[0]
        num_vc = priors.shape[1]

        grasp_priors_xdim = priors.shape[-2]
        grasp_priors_ydim = priors.shape[-1]

        independent_finger_pos_xdim = independent.shape[0]
        independent_finger_pos_ydim = independent.shape[1]

        x_border = (independent_finger_pos_xdim-grasp_priors_xdim)/2
        y_border = (independent_finger_pos_ydim-grasp_priors_ydim)/2

        out = np.zeros((num_heatmaps, grasp_priors_xdim, grasp_priors_ydim))

        for current_heatmap_index in range(num_heatmaps):

            independent_finger_pos = independent[:, :, current_heatmap_index]
            independent_finger_pos_cropped = independent_finger_pos[x_border:-x_border-1, y_border:-y_border-1]

            out[current_heatmap_index] = independent_finger_pos_cropped

            vc_index_for_heatmap = current_heatmap_index % num_vc

            # import IPython
            # IPython.embed()
            for j in range(num_vc):

                out[current_heatmap_index] *= priors[current_heatmap_index, j, vc_index_for_heatmap]


        return out







class MultiplyPriors(ClassificationStage):

    def __init__(self):
        pass

    def dataset_inited(self, dataset):
        return 'independent_x_priors' in dataset.keys()

    def init_dataset(self, dataset):
        independent_x_priors = self._run(dataset, 0 )

        shape = (900, independent_x_priors[0], independent_x_priors[1], independent_x_priors[2])
        dataset.create_dataset('independent_x_priors', shape)

    def _run(self, dataset, index):
        priors = dataset['convolved_heatmaps'][index]
        independent = dataset['rescaled_heatmaps'][index]

        num_heatmaps = independent.shape[-1]
        for i in range(num_heatmaps):

            #grasp_priors shape (9,253,413)
            grasp_priors = priors[0]

            #independent_finger_pos.shape (352, 512)
            independent_finger_pos = independent[i]

            x_border = (independent_finger_pos[0]-grasp_priors[1])/2
            y_border = (independent_finger_pos[1]-grasp_priors[2])/2

            independent_finger_pos_cropped = independent_finger_pos[x_border:-x_border-1, y_border:-y_border-1]

            num_prior_maps = grasp_priors.shape[0]
            for j in range(num_prior_maps):
                #out = independent_finger_pos_cropped *
                #  Need to multiply independent by the priors for the other fingers.
                print 'not done.'


class GetTopNGrasps(ClassificationStage):

    def __init__(self, mask=None):
        self.grasps = []
        self.mask = mask
        self.x_border = 0
        self.y_border = 0

        self.out_key = 'top_n_grasps'

    def dataset_inited(self, dataset):
        return True

    def init_dataset(self, dataset):
        pass

    def _run(self, dataset, index):
        self.grasps = []
        independent_x_priors = dataset['independent_x_priors'][index]

        num_heatmaps = independent_x_priors.shape[0]
        num_grasp_types = num_heatmaps/4

        for i in range(num_grasp_types):
            palm_index = i*4
            independent_x_priors_image = np.copy(independent_x_priors[palm_index])
            heatmaps = np.copy(independent_x_priors[palm_index:palm_index+4])


            if self.mask is not None:
                x_dim, y_dim = independent_x_priors[palm_index].shape
                mask_x_dim , mask_y_dim = self.mask.shape

                if x_dim != mask_x_dim:
                    self.x_border = (mask_x_dim-x_dim)/2
                    self.y_border = (mask_y_dim-y_dim)/2

                    x_offset = 0
                    y_offset = 0
                    if x_dim % 2 ==1:
                        x_offset = 1
                    if y_dim % 2 == 1:
                        y_offset = 1
                    self.mask = self.mask[self.x_border:-self.x_border-x_offset, self.y_border:-self.y_border-y_offset]

                independent_x_priors_image = independent_x_priors[palm_index] * self.mask

            argmax_v,argmax_u = np.unravel_index(independent_x_priors_image.argmax(), independent_x_priors_image.shape)

            grasp_energy = independent_x_priors_image[argmax_v, argmax_u]
            grasp_type = i
            self.grasps.append((grasp_energy, grasp_type, palm_index, argmax_v, argmax_u, self.x_border, self.y_border, heatmaps))

        self.grasps.sort(reverse=True)

    def run(self, dataset, index):
        self._run(dataset, index)



class RefineGrasps(ClassificationStage):

    def __init__(self, grasps):
        self.out_key = 'refined_grasps'

        self.grasps = grasps
        self.refined_grasps = []

    def dataset_inited(self, dataset):
        return True

    def init_dataset(self, dataset):
        pass

    def _run(self, dataset, index):
        self.grasps = []
        independent_x_priors = dataset['independent_x_priors'][index]

        num_heatmaps = independent_x_priors.shape[0]
        num_grasp_types = num_heatmaps/4

        for i in range(num_grasp_types):
            # self.grasps.append((grasp_energy, grasp_type, palm_index, argmax_v+self.x_border, argmax_u+self.y_border))
            grasp = self.grasps[i]

            palm_index = i*4
            independent_x_priors_palm_image = independent_x_priors[palm_index]
            independent_x_priors_f1_image = independent_x_priors[palm_index + 1]
            independent_x_priors_f2_image = independent_x_priors[palm_index + 2]
            independent_x_priors_f3image = independent_x_priors[palm_index + 3]






    def run(self, dataset, index):
        self._run(dataset, index)








