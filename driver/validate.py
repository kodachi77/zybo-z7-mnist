import argparse
import gzip
import os
import numpy as np
from mnist_overlay import MNISTOverlay, io_shape_dict


def load_mnist_data(dataset_root, one_hot=False):
    """
    Loads the MNIST data from gzipped files in the given dataset root folder.

    Parameters:
    -----------
    dataset_root (str): Path to the folder containing gzipped MNIST files.
    one_hot (bool): If True, convert labels to one-hot encoding.

    Returns:
        tuple: trainx, trainy, testx, testy, valx, valy arrays.
    """

    def load_images(file_path):
        with gzip.open(file_path, "rb") as f:
            _ = int.from_bytes(f.read(4), "big")  # Magic number
            num_images = int.from_bytes(f.read(4), "big")
            rows = int.from_bytes(f.read(4), "big")
            cols = int.from_bytes(f.read(4), "big")
            data = np.frombuffer(f.read(), dtype=np.uint8)
            return data.reshape(num_images, rows * cols)  # no normalization / 255.0

    def load_labels(file_path):
        with gzip.open(file_path, "rb") as f:
            _ = int.from_bytes(f.read(4), "big")  # Magic number
            num_labels = int.from_bytes(f.read(4), "big")
            labels = np.frombuffer(f.read(), dtype=np.uint8)
            return labels

    def to_one_hot(labels, num_classes=10):
        return np.eye(num_classes)[labels]

    # Paths to MNIST gzipped data files
    train_images_path = os.path.join(dataset_root, "train-images-idx3-ubyte.gz")
    train_labels_path = os.path.join(dataset_root, "train-labels-idx1-ubyte.gz")
    test_images_path = os.path.join(dataset_root, "t10k-images-idx3-ubyte.gz")
    test_labels_path = os.path.join(dataset_root, "t10k-labels-idx1-ubyte.gz")

    # Load data
    trainx = load_images(train_images_path)
    trainy = load_labels(train_labels_path)
    testx = load_images(test_images_path)
    testy = load_labels(test_labels_path)

    # Convert to one-hot encoding if required
    if one_hot:
        trainy = to_one_hot(trainy)
        testy = to_one_hot(testy)

    # Split train data into train and validation sets
    val_size = int(0.1 * len(trainx))  # Use 10% of the training data for validation
    valx, valy = trainx[:val_size], trainy[:val_size]
    trainx, trainy = trainx[val_size:], trainy[val_size:]

    return trainx, trainy, testx, testy, valx, valy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate top-1 accuracy of MNIST accelerator"
    )
    parser.add_argument(
        "--batchsize", help="number of samples for inference", type=int, default=1000
    )

    parser.add_argument(
        "--bitfile", help='name of bitfile (i.e. "resizer.bit")', default="resizer.bit"
    )

    # parse arguments
    args = parser.parse_args()
    batch_size = args.batchsize
    bitfile = args.bitfile

    trainx, trainy, testx, testy, valx, valy = load_mnist_data("data", False)

    test_imgs = testx
    test_labels = testy

    ok = 0
    nok = 0
    total = test_imgs.shape[0]

    driver = MNISTOverlay(bitfile_name=bitfile, batch_size=batch_size)

    n_batches = int(total / batch_size)

    test_imgs = test_imgs.reshape(n_batches, batch_size, -1)
    test_labels = test_labels.reshape(n_batches, batch_size)

    for i in range(n_batches):
        ibuf_normal = test_imgs[i].reshape(driver.ibuf_packed_device[0].shape)
        exp = test_labels[i]
        driver.copy_input_data_to_device(ibuf_normal)
        driver.execute_on_buffers()
        obuf_normal = np.empty_like(driver.obuf_packed_device[0])
        driver.copy_output_data_from_device(obuf_normal)
        ret = np.bincount(obuf_normal.flatten() == exp.flatten())
        nok += ret[0]
        ok += ret[1]
        print("batch %d / %d : total OK %d NOK %d" % (i + 1, n_batches, ok, nok))

    acc = 100.0 * ok / (total)
    print("Final accuracy: %f" % acc)
