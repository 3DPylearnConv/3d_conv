import h5py
from datasets.reconstruction_dataset import ReconstructionDataset, build_training_example
from multiprocessing import Pool

PATCH_SIZE = 24
OUT_FILE_PATH = "22_model_big_bird_1000_rot_24x24x24.h5"

from multiprocessing import Process, Queue

def reader(index_queue, examples_queue):

    while True:
        msg = index_queue.get()
        if (msg == 'DONE'):
            break
        else:
            index = msg
            single_view_pointcloud_filepath = recon_dataset.examples[index][0]
            pose_filepath = recon_dataset.examples[index][1]
            model_filepath = recon_dataset.examples[index][2]
            try:
                x, y = build_training_example(model_filepath, pose_filepath, single_view_pointcloud_filepath, PATCH_SIZE)
                examples_queue.put((index, x, y))
            except:
                examples_queue.put((index, None, None))

if __name__=='__main__':

    recon_dataset = ReconstructionDataset(patch_size=PATCH_SIZE)
    num_examples = recon_dataset.get_num_examples()

    with h5py.File(OUT_FILE_PATH, 'w') as h5_dset:

        h5_dset.create_dataset('x', (num_examples, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, 1), chunks=(100, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, 1))
        h5_dset.create_dataset('y', (num_examples, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, 1), chunks=(100, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, 1))

        index_queue = Queue()
        examples_queue = Queue(maxsize=100)

        print("staring readers")
        num_readers = 6
        for i in range(num_readers):
            reader_p = Process(target=reader, args=(index_queue, examples_queue))
            reader_p.daemon = True
            reader_p.start()

        print("putting indices on queue")
        for i in range(num_examples):
            index_queue.put(i)

        print("putting done statments on queue")
        for i in range(num_readers):
            index_queue.put('DONE')

        print("staring to write examples to h5dset")
        for i in range(num_examples):

            if i % 1000 == 0:
                print("working on number: " + str(i))
            index, x, y = examples_queue.get()

            if x is None or y is None:
                print("skipping index: " + str(index))
                continue

            h5_dset['x'][index] = x
            h5_dset['y'][index] = y



