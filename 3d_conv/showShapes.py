

from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import sys
from datasets.model_net_dataset import ModelNetDataset
from layers.layer_utils import *
from layers.layer_utils import downscale_3d



def showShapes(num):

    models_dir = '/srv/3d_conv_data/ModelNet10'
    patch_size = 256
    downsample_factor=16


    train_dataset = ModelNetDataset(models_dir, patch_size, dataset_type='train')
    categories = train_dataset.get_categories()



    train_iterator = train_dataset.iterator(batch_size=num,
                                                    num_batches=1,
                                                    mode='even_shuffled_sequential', type='classify')
    mini_batch_x, mini_batch_y = train_iterator.next(categories)

    mini_batch_x = downscale_3d(mini_batch_x, downsample_factor)
    dimension = 16

    for example in xrange(num):


        toPlot = mini_batch_x[example]
        toPlot = toPlot.reshape(16,16,16)
        x,y,z = toPlot.nonzero()
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')

        ax.plot([0],[0],[0],'w')
        ax.plot([dimension],[dimension],[dimension],'w')


        ax.scatter(x,y,z, c= 'blue')


        ax.set_zlim3d(0,dimension)
        ax.set_xlim3d(0,dimension)
        ax.set_ylim3d(0,dimension)
        print categories[mini_batch_y[example]]

        plt.show()





if __name__ == '__main__':
    showShapes(40)
