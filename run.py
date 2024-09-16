import tensorflow as tf
import numpy as np
from model1 import getModel,real2complex
from losses import mae, mse
from data import load_batch,load_test
import matplotlib.pyplot as plt
import os
import random
from scipy.io import savemat

num_file = 2
EPOCHS = 50
BATCH_SIZE = 40
nb_train = 20000
nb_val = 1000
regularization_rate = 1e-7
data_path = '/data4/ly/Saber_denoise/'
#model_save_path = './model-125-25'
model_save_path = './test_model1'
model_name = 'model.ckpt'
lr_base = 0.001
lr_decay_rate = 0.95
loss={'batch':[], 'count':[], 'epoch':[]}
val_loss=[]
num_FDN = 1
num_SDN = 4
y_axis = 1
x_axis = 8192
os.environ["CUDA_VISIBLE_DEVICES"]="7"
import time

def complex2real(x):
    x_real = np.real(x)
    x_imag = np.imag(x)
    return np.concatenate([x_real,x_imag], axis=-1)

def real2complex_array(x):
    x = np.asarray(x)
    channel = x.shape[-1] // 2
    x_real = x[:,:,:,:channel]
    x_imag = x[:,:,:,channel:]
    return x_real + x_imag * 1j

def train():
    # nb_train = len(train_data)
    x = tf.placeholder(tf.float32,shape=(BATCH_SIZE,y_axis,x_axis,2),name='x_input')
    y_ = tf.placeholder(tf.float32,shape=(BATCH_SIZE,y_axis,x_axis,2),name='y_label')

    y_SDN1 = getModel(x)
    global_step = tf.Variable(0.,trainable=False)
    with tf.name_scope('mse_loss'):
        part1_loss = y_SDN1[:,:,5000:5200,:]
        part2_loss = y_SDN1[:,:,7160:7280,:]
        part1_label = y_[:,:,5000:5200,:]
        part2_label = y_[:,:,7160:7280,:]
        total_loss = mae(y_, y_SDN1) + 2*mae(part1_label,part1_loss) + 2*mae(part2_label,part2_loss)
    lr = tf.train.exponential_decay(lr_base,
                                    global_step=global_step,
                                    decay_steps=20000,
                                    decay_rate=lr_decay_rate,
                                    staircase=False)
    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(lr).minimize(total_loss,global_step=global_step)

    saver = tf.train.Saver(max_to_keep=40)
    with tf.Session() as sess:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        # saver.restore(sess, os.path.join(model_save_path, 'model.ckpt-512500'))
        label_all, input_all = load_batch(nb_train, data_path, num_file)
        index = [i for i in range(nb_train)]
        random.shuffle(index)
        input_all = input_all[index]
        label_all = label_all[index]
        train_loss = []
        val_loss = []
        for i in range(EPOCHS):
            loss_sum = 0.0
            count_batch = 0

            nb_batches = int((nb_train) // BATCH_SIZE)
            for n_batch in range(nb_batches):
                input_data_batch = input_all[n_batch * BATCH_SIZE:(n_batch + 1) * BATCH_SIZE, :, :, :]
                label_batch = label_all[n_batch * BATCH_SIZE:(n_batch + 1) * BATCH_SIZE, :, :, :]
                _,loss_value, step = sess.run([train_step, total_loss, global_step],
                                              feed_dict={x: input_data_batch, y_: label_batch})

                loss_sum += loss_value
                count_batch += 1
                ave_loss = loss_sum / count_batch
                loss['batch'].append(ave_loss)
                print ('Epoch %3d-batch %3d/%3d  training loss: %8f ' % (i+1, count_batch, nb_batches, ave_loss))

                # evaluate
            train_loss.append(ave_loss)
            saver.save(sess, os.path.join(model_save_path,model_name), global_step=global_step)
            label_test, input_test = load_test(nb_val, data_path)
            nb_batches = int((nb_val) // BATCH_SIZE)
            loss_sum = 0.0
            count_batch = 0
            for k in range(nb_batches):
                input_data_batch = input_test[k * BATCH_SIZE:(k + 1) * BATCH_SIZE, :, :, :]
                label_batch = label_test[k * BATCH_SIZE:(k + 1) * BATCH_SIZE, :, :, :]
                loss_value = sess.run(total_loss,
                                               feed_dict={x: input_data_batch, y_: label_batch})

                loss_sum += loss_value
                count_batch += 1
                ave_loss = loss_sum / count_batch
                loss['batch'].append(ave_loss)
                print('Epoch %3d-batch %3d/%3d  val loss: %8f ' % (i + 1, count_batch, nb_batches, ave_loss))
            val_loss.append(ave_loss)
            # test every 5 epochs
        savemat("/data1/ly/matlab_project/Saber_denoise/trainloss.mat", {'trainloss': train_loss})
        savemat("/data1/ly/matlab_project/Saber_denoise/valloss.mat", {'valloss': val_loss})


def main(argv=None):
    start_time = time.time()
    train()
    end_time = time.time()
    print(f"运行时间: {end_time - start_time} 秒")

if __name__ == '__main__':
    tf.app.run()