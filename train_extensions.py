from pylearn2.train_extensions import TrainExtension
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import os

import pylearn2.utils.serial


class RecordWeights(TrainExtension):

    def __init__(self, save_path, skip_num=1):
        self.save_path = save_path
        self.skip_num = skip_num

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        self.current_weight_file_number = 0
        self.current_iteration = 0

    def on_monitor(self, model, dataset, algorithm):
        if self.current_iteration % self.skip_num == 0:
            self.plot(model)

        self.current_iteration += 1

    def plot(self, model):

        #get the current weights from the model
        weights = model.get_weights_topo()

        num_weights = weights.shape[0]
        num_channels = weights.shape[-1]

        #we are going to plot every channel for each kernel
        num_plots = num_weights*num_channels

        #set up the dimensions of the figure
        w = math.ceil(math.sqrt(num_plots))
        h = w
        
        subplot_count = 1
        plt.figure()
        for i in range(num_weights):
            for j in range(num_channels):
                # add a new subplot for each kernel
                sub = plt.subplot(h, w, subplot_count)
                #remove the axis so it looks prettier
                sub.axes.get_xaxis().set_visible(False)
                sub.axes.get_yaxis().set_visible(False)
                #make it greyscale
                plt.imshow(weights[i, :, :, j], cmap=cm.Greys_r)
                subplot_count += 1

        plt.savefig(self.save_path + 'weight_' + str(self.current_weight_file_number) + '.png')

        #must close this or else plt leaks
        plt.close()

        self.current_weight_file_number += 1


class RecordCurrentModel(TrainExtension):

    def __init__(self, save_path):
        self.save_path = save_path
        self.save_count = 0

    def on_monitor(self, model, dataset, algorithm):
        if self.save_count % 10 == 0:
            pylearn2.utils.serial.save(self.save_path, model, on_overwrite='backup')
        self.save_count += 1

