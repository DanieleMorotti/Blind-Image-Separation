import numpy as np

from tensorflow.keras.datasets import mnist, fashion_mnist
from sklearn.model_selection import train_test_split


# Download mnist and fashion mnist datasets and normalize values 
def download_datasets():
    print("Downloading mnist and fashion mnist datasets ...\n")
    (mnist_x_train, _), (mnist_x_test, _) = mnist.load_data()
    mnist_x_train, mnist_x_val = train_test_split(mnist_x_train, test_size=0.15)

    (fashion_mnist_x_train, _), (fashion_mnist_x_test, _) = fashion_mnist.load_data()
    fashion_mnist_x_train, fashion_mnist_x_val = train_test_split(fashion_mnist_x_train, test_size=0.15)

    # Pad and normalize images
    mnist_x_train = np.pad(mnist_x_train,((0,0),(2,2),(2,2)))/255.
    mnist_x_val = np.pad(mnist_x_val, ((0,0),(2,2),(2,2)))/255.
    mnist_x_test = np.pad(mnist_x_test,((0,0),(2,2),(2,2)))/255.

    fashion_mnist_x_train = np.pad(fashion_mnist_x_train,((0,0),(2,2),(2,2)))/255.
    fashion_mnist_x_val = np.pad(fashion_mnist_x_val,((0,0),(2,2),(2,2)))/255.
    fashion_mnist_x_test = np.pad(fashion_mnist_x_test,((0,0),(2,2),(2,2)))/255.

    print(f"The train datasets have the following shape: {mnist_x_train.shape}.")
    print(f"The validation datasets have the following shape: {mnist_x_val.shape}.")
    print(f"The test datasets have the following shape: {mnist_x_test.shape}.\n")

    return (mnist_x_train, mnist_x_val, mnist_x_test), (fashion_mnist_x_train, fashion_mnist_x_val, fashion_mnist_x_test) 


# Function that sums together images from the 2 different datasets 
def datagenerator(x1, x2, batchsize):
    n1 = x1.shape[0]
    n2 = x2.shape[0]
    while True:
        num1 = np.random.randint(0, n1, batchsize)
        num2 = np.random.randint(0, n2, batchsize)

        x_data = (x1[num1] + x2[num2]) / 2.0
        y_data = np.concatenate((x1[num1], x2[num2]), axis=2)

        yield x_data, y_data 


def create_generators(batch_train=32, batch_val=32, batch_test=1024):
    '''
    This function returns the 3 generators for training, validation and test sets.
    Args:
        m_train: (object) the mnist train dataset.
        fm_train: (object) the fashion mnist train dataset.
        ...
        bs_train: (int) the batch size to use for train generator.
        bs_val: (int) the batch size to use for validation generator.
        bs_test: (int) the batch size to use for test generator.
    '''
    print("DATASET GENERATION\n")
    (m_train, m_val, m_test), (fm_train, fm_val, fm_test) = download_datasets()

    train_generator = datagenerator(m_train, fm_train, batch_train)
    val_generator = datagenerator(m_val, fm_val, batch_val)
    test_generator = datagenerator(m_test, fm_test, batch_test)

    return train_generator, val_generator, test_generator