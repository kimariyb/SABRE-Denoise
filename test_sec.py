import numpy as np
import tensorflow as tf
from model1 import getModel
from data import load_batch
from scipy.io import savemat
import os

data_type = 'temp'
num_FDN = 1
num_SDN = 4
y_axis = 1
nb_train = 1
num_file = 1
import time

if data_type == 'temp':
    x_axis = 8192
    data_path = '/data4/ly/Saber_denoise/test_data/'
    BATCH_SIZE = 1

model_path = './model1'

os.environ["CUDA_VISIBLE_DEVICES"]="7"
regularization_rate = 1e-7
def test():
    x = tf.placeholder(tf.float32, shape=(BATCH_SIZE, y_axis, x_axis, 2), name='x_input')
    y_ = tf.placeholder(tf.float32, shape=(BATCH_SIZE, y_axis, x_axis, 2), name='y_label')

    y_SDN1 = getModel(x)
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)
        saver.restore(sess, os.path.join(model_path, 'model.ckpt-5500'))
        start_time = time.time()
        label_all, input_all = load_batch(nb_train, data_path, num_file)
        predict= sess.run([y_SDN1],feed_dict={x: input_all})

        predict = predict[0]

        pred_c = predict[:,:,:,0] + 1j*predict[:,:,:,1]
        out = np.zeros_like(pred_c)
        out[:,:,5040:5140] = pred_c[:,:,5040:5140]
        out[:,:,7050:7300] = pred_c[:,:,7050:7300]


        savemat("/data1/ly/matlab_project/Saber_denoise/denoise.mat", {'F': out})
        end_time = time.time()
        print(f"运行时间: {end_time - start_time} 秒")






test()
