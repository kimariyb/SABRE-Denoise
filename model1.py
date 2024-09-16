import tensorflow  as tf
import numpy as np
from tensorflow.python.ops.init_ops import Initializer



class ComplexInit(Initializer):

    def __init__(self, kernel_size, input_dim,
                 weight_dim, nb_filters=None,
                 criterion='glorot', seed=None):

        # `weight_dim` is used as a parameter for sanity check
        # as we should not pass an integer as kernel_size when
        # the weight dimension is >= 2.
        # nb_filters == 0 if weights are not convolutional (matrix instead of filters)
        # then in such a case, weight_dim = 2.
        # (in case of 2D input):
        #     nb_filters == None and len(kernel_size) == 2 and_weight_dim == 2
        # conv1D: len(kernel_size) == 1 and weight_dim == 1
        # conv2D: len(kernel_size) == 2 and weight_dim == 2
        # conv3d: len(kernel_size) == 3 and weight_dim == 3

        assert len(kernel_size) == weight_dim and weight_dim in {0, 1, 2, 3}
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.input_dim = input_dim
        self.weight_dim = weight_dim
        self.criterion = criterion
        self.seed = 1337 if seed is None else seed

    def __call__(self, shape, dtype=None, partition_info=None):

        if self.nb_filters is not None:
            kernel_shape = tuple(self.kernel_size) + (int(self.input_dim), self.nb_filters)
        else:
            kernel_shape = (int(self.input_dim), self.kernel_size[-1])

        fan_in, fan_out = _compute_fans(
            tuple(self.kernel_size) + (self.input_dim, self.nb_filters)
        )

        if self.criterion == 'glorot':
            s = 1. / (fan_in + fan_out)
        elif self.criterion == 'he':
            s = 1. / fan_in
        else:
            raise ValueError('Invalid criterion: ' + self.criterion)
        rng = RandomState(self.seed)
        modulus = rng.rayleigh(scale=s, size=kernel_shape)
        phase = rng.uniform(low=-np.pi, high=np.pi, size=kernel_shape)
        weight_real = modulus * np.cos(phase)
        weight_imag = modulus * np.sin(phase)
        weight = np.concatenate([weight_real, weight_imag], axis=-1)

        return weight



def Fully_connected(x, units, layer_name) :
    with tf.name_scope(layer_name) :
        return tf.layers.dense(inputs=x, use_bias=True, units=units)


def complex_conv2d(input,name,kw=3,kh=3,n_out=32,sw=1,sh=1,activation=True):
    n_in = input.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel_init = ComplexInit(kernel_size=(kh,kw),
                                  input_dim=n_in,
                                  weight_dim=2,
                                  nb_filters=n_out,
                                  criterion='he')
        kernel = tf.get_variable(scope + 'weights',
                                 shape=[kh,kw,n_in,n_out],
                                 dtype=tf.float32,
                                 initializer=kernel_init)
        bias_init = tf.constant(0.0001,dtype=tf.float32,shape=[n_out*2])
        biases = tf.get_variable(scope+'biases', dtype=tf.float32, initializer=bias_init)

        conv = tf.nn.conv2d(input,kernel,strides=[1,sh,sw,1],padding='SAME')
        conv_bias = tf.nn.bias_add(conv,biases)
        if activation:
            act = tf.nn.leaky_relu(conv_bias)
            output = act

        else:
            output = conv_bias


        return output

def Conv_transpose(x, name,filter_size, in_filters, out_filters, fraction=2, padding="SAME"):
    with tf.variable_scope(name):
        n = filter_size * filter_size * out_filters
        kernel = tf.get_variable('filter', [1, filter_size, out_filters, in_filters], tf.float32,
                                 initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / n)))
        size = tf.shape(x)
        output_shape = tf.stack([size[0], size[1], size[2] * fraction, out_filters])
        x = tf.nn.conv2d_transpose(x, kernel, output_shape, [1, fraction, fraction, 1], padding)

        return x

def real2complex(x):
    channel = x.shape[-1] // 2
    if x.shape.ndims == 3:
        return tf.complex(x[:,:,:channel], x[:,:,channel:])
    elif x.shape.ndims == 4:
        return tf.complex(x[:,:,:,:channel], x[:,:,:,channel:])

def complex2real(x):
    x_real = np.real(x)
    x_imag = np.imag(x)
    return np.concatenate([x_real,x_imag], axis=-1)

def dc(generated, X_k, mask):
    gene_complex = real2complex(generated)
    gene_complex = tf.transpose(gene_complex, [0, 3, 1, 2])
    mask = tf.transpose(mask, [0, 3, 1, 2])
    X_k = tf.transpose(X_k, [0, 3, 1, 2])
    gene_fft = tf.ifft2d(gene_complex)
    out_fft = X_k + gene_fft * (1.0 - mask)
    output_complex = tf.fft2d(out_fft)
    output_complex = tf.transpose(output_complex, [0, 2, 3, 1])
    output_real = tf.cast(tf.real(output_complex), dtype=tf.float32)
    output_imag = tf.cast(tf.imag(output_complex), dtype=tf.float32)
    output = tf.concat([output_real, output_imag], axis=-1)
    return output

def dc_tdomain(generated, X_k, mask):
    gene_t = real2complex(generated)
    out_fft = X_k + gene_t * (1.0 - mask)
    output_complex = out_fft
    output_real = tf.cast(tf.real(output_complex), dtype=tf.float32)
    output_imag = tf.cast(tf.imag(output_complex), dtype=tf.float32)
    output = tf.concat([output_real,output_imag], axis=-1)
    return output

def complex_real(x):
    output_real = tf.cast(tf.real(x), dtype=tf.float32)
    output_imag = tf.cast(tf.imag(x), dtype=tf.float32)
    output = tf.concat([output_real, output_imag], axis=-1)
    return output
#X_K是时域采样后的数据

def getModel(x):
    temp_1 = x
    conv1_SDN1 = complex_conv2d(temp_1, 'conv1_SDN1', kw=3, kh=1, n_out=16, sw=2, sh=1, activation=True)
    conv2_SDN1 = complex_conv2d(conv1_SDN1, 'conv2_SDN1', kw=3, kh=1, n_out=32, sw=2, sh=1, activation=True)
    conv3_SDN1 = complex_conv2d(conv2_SDN1, 'conv3_SDN1', kw=3, kh=1, n_out=64, sw=2, sh=1, activation=True)
    conv4_SDN1 = complex_conv2d(conv3_SDN1, 'conv4_SDN1', kw=3, kh=1, n_out=128, sw=2, sh=1, activation=True)
    conv5_SDN1 = complex_conv2d(conv4_SDN1, 'conv5_SDN1', kw=3, kh=1, n_out=256, sw=2, sh=1, activation=True)
    deconv5_SDN1 = Conv_transpose(conv5_SDN1, 'deconv5_SDN1', filter_size=3, in_filters=512, out_filters=256)
    SDN1_up4 = tf.concat([deconv5_SDN1, conv4_SDN1], axis=3, name='SDN1_up4')
    deconv4_SDN1 = Conv_transpose(SDN1_up4, 'deconv4_SDN1', filter_size=3, in_filters=512, out_filters=128)
    SDN1_up3 = tf.concat([deconv4_SDN1, conv3_SDN1], axis=3, name='SDN1_up3')
    deconv3_SDN1 = Conv_transpose(SDN1_up3, 'deconv3_SDN1', filter_size=3, in_filters=256, out_filters=64)
    SDN1_up2 = tf.concat([deconv3_SDN1, conv2_SDN1], axis=3, name='SDN1_up2')
    deconv2_SDN1 = Conv_transpose(SDN1_up2, 'deconv2_SDN1', filter_size=3, in_filters=128, out_filters=32)
    SDN1_up1 = tf.concat([deconv2_SDN1, conv1_SDN1], axis=3, name='SDN1_up1')
    deconv1_SDN1 = Conv_transpose(SDN1_up1, 'deconv1_SDN1', filter_size=3, in_filters=64, out_filters=2)
    block_SDN1 = deconv1_SDN1 + temp_1

    conv1_SDN2 = complex_conv2d(block_SDN1, 'conv1_SDN2', kw=3, kh=1, n_out=16, sw=2, sh=1, activation=True)
    conv2_SDN2 = complex_conv2d(conv1_SDN2, 'conv2_SDN2', kw=3, kh=1, n_out=32, sw=2, sh=1, activation=True)
    conv3_SDN2 = complex_conv2d(conv2_SDN2, 'conv3_SDN2', kw=3, kh=1, n_out=64, sw=2, sh=1, activation=True)
    conv4_SDN2 = complex_conv2d(conv3_SDN2, 'conv4_SDN2', kw=3, kh=1, n_out=128, sw=2, sh=1, activation=True)
    conv5_SDN2 = complex_conv2d(conv4_SDN2, 'conv5_SDN2', kw=3, kh=1, n_out=256, sw=2, sh=1, activation=True)
    deconv5_SDN2 = Conv_transpose(conv5_SDN2, 'deconv5_SDN2', filter_size=3, in_filters=512, out_filters=256)
    SDN2_up4 = tf.concat([deconv5_SDN2, conv4_SDN2], axis=3, name='SDN2_up4')
    deconv4_SDN2 = Conv_transpose(SDN2_up4, 'deconv4_SDN2', filter_size=3, in_filters=512, out_filters=128)
    SDN2_up3 = tf.concat([deconv4_SDN2, conv3_SDN2], axis=3, name='SDN2_up3')
    deconv3_SDN2 = Conv_transpose(SDN2_up3, 'deconv3_SDN2', filter_size=3, in_filters=256, out_filters=64)
    SDN2_up2 = tf.concat([deconv3_SDN2, conv2_SDN2], axis=3, name='SDN2_up2')
    deconv2_SDN2 = Conv_transpose(SDN2_up2, 'deconv2_SDN2', filter_size=3, in_filters=128, out_filters=32)
    SDN2_up1 = tf.concat([deconv2_SDN2, conv1_SDN2], axis=3, name='SDN2_up1')
    deconv1_SDN2 = Conv_transpose(SDN2_up1, 'deconv1_SDN2', filter_size=3, in_filters=64, out_filters=2)

    conv1_out1 = complex_conv2d(deconv1_SDN2, 'conv_final1', kw=3, kh=1, n_out=16, sw=1, sh=1, activation=True)
    conv1_out2 = complex_conv2d(conv1_out1, 'conv_final2', kw=3, kh=1, n_out=1, sw=1, sh=1, activation=True)


    return conv1_out2