import h5py
import numpy    as np
import scipy.io as sio
import tensorflow as tf
import os
import random
import matplotlib.pyplot as plt
from model1 import real2complex, complex2real


y_axis = 1
x_aixs = 8192


def gene_full_data(x):
    real_data = x[:,:,:, 0]
    imag_data = x[:,:,:, 1]
    full_data = real_data + 1j * imag_data
    return full_data

def load_batch(nb_train, path, num_file):
    input_all = []
    label_all = []
    for i in range(num_file):
        full_data = sio.loadmat(path + '/label' + str(i+1) + '.mat')
        impure_data = sio.loadmat(path + '/input' + str(i+1) + '.mat')
        impure_data = impure_data['input_data']
        full_data = full_data['label_data']
        impure_data = np.expand_dims(impure_data, axis=1)
        full_data = np.expand_dims(full_data, axis=1)
        input_all.append(impure_data)
        label_all.append(full_data)
        print('load data No.%d'%(int(i)))
    input_all = np.reshape(input_all, [nb_train, y_axis, x_aixs, 2])
    label_all = np.reshape(label_all, [nb_train, y_axis, x_aixs, 2])
    return label_all, input_all

def load_test(nb_train, path):
    input_all = []
    label_all = []
    full_data = sio.loadmat(path + '/test_label.mat')
    impure_data = sio.loadmat(path + '/test_input.mat')
    impure_data = impure_data['input_data']
    full_data = full_data['label_data']
    impure_data = np.expand_dims(impure_data, axis=1)
    full_data = np.expand_dims(full_data, axis=1)
    input_all.append(impure_data)
    label_all.append(full_data)
    print('load data test data')
    input_all = np.reshape(input_all, [nb_train, y_axis, x_aixs, 2])
    label_all = np.reshape(label_all, [nb_train, y_axis, x_aixs, 2])
    return label_all, input_all

