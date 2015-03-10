
import theano
import pylearn2

import numpy as np
import h5py

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







