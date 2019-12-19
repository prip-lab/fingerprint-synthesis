import glob
import numpy as np
import os, sys
import argparse

from tensorpack import *
from tensorpack.utils.viz import *
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.utils.globvars import globalns as opt
import tensorflow as tf
from tensorpack.tfutils.common import get_tensors_by_names
from GAN import GANModelDesc
import timeit
import imageio
import skimage

opt.SHAPE = 512
opt.BATCH = 32 
opt.Z_DIM = 512 


class Model(GANModelDesc):
    def _get_inputs(self):
        return [InputDesc(tf.float32, (None, opt.SHAPE, opt.SHAPE, 3), 'input')]

    def generator(self, z):
        nf = 16
        l = FullyConnected('fc0', z, nf * 64 * 4 * 4, nl=tf.identity)
        l = tf.reshape(l, [-1, 4, 4, nf * 64])
        l = BNReLU(l)
        with argscope(Deconv2D, nl=BNReLU, kernel_shape=4, stride=2):
            l = Deconv2D('deconv1', l, [8, 8, nf * 32])
            l = Deconv2D('deconv2', l, [16, 16, nf * 16])
            l = Deconv2D('deconv3', l, [32, 32, nf*8])
            l = Deconv2D('deconv4', l, [64, 64, nf * 4])
            l = Deconv2D('deconv5', l, [128, 128, nf * 2])
            l = Deconv2D('deconv6', l, [256, 256, nf * 1])
            l = Deconv2D('deconv7', l, [512, 512, 3], nl=tf.identity)
            l = tf.tanh(l, name='gen')
        return l

    @auto_reuse_variable_scope
    def discriminator(self, imgs):
        nf = 16
        with argscope(Conv2D, nl=tf.identity, kernel_shape=4, stride=2), \
                argscope(LeakyReLU, alpha=0.2):
            l = (LinearWrap(imgs)
                 .Conv2D('conv0', nf, nl=LeakyReLU)
                 .Conv2D('conv1', nf * 2)
                 .BatchNorm('bn1').LeakyReLU()
                 .Conv2D('conv2', nf * 4)
                 .BatchNorm('bn2').LeakyReLU()
                 .Conv2D('conv3', nf * 8)
                 .BatchNorm('bn3').LeakyReLU()
                 .Conv2D('conv4', nf * 16)
                 .BatchNorm('bn4').LeakyReLU()
                 .Conv2D('conv5', nf * 32)
                 .BatchNorm('bn5').LeakyReLU()
                 .Conv2D('conv6', nf * 64)
                 .BatchNorm('bn6').LeakyReLU()
                 .FullyConnected('fct', 1, nl=tf.identity)())
        return l

    def _build_graph(self, inputs):
        image_pos = inputs[0]
        image_pos = image_pos / 128.0 - 1

        z = tf.random_uniform([opt.BATCH, opt.Z_DIM], -1, 1, name='z_train')
        z = tf.placeholder_with_default(z, [None, opt.Z_DIM], name='z')

        with argscope([Conv2D, Deconv2D, FullyConnected],
                      W_init=tf.truncated_normal_initializer(stddev=0.02)):
            with tf.variable_scope('gen'):
                image_gen = self.generator(z)
            tf.summary.image('generated-samples', image_gen, max_outputs=30)
            with tf.variable_scope('discrim'):
                vecpos = self.discriminator(image_pos)
                vecneg = self.discriminator(image_gen)

        self.build_losses(vecpos, vecneg)
        self.collect_variables()

    def _get_optimizer(self):
        lr = symbolic_functions.get_scalar_var('learning_rate', 2e-4, summary=True)
        return tf.train.AdamOptimizer(lr, beta1=0.5, epsilon=1e-3)

def sample2(model, model_path,sample_path, num, output_name='gen/gen'):
    config = PredictConfig(
        session_init=get_model_loader(model_path),
        model=model,
        input_names=['z'],
        output_names=[output_name, 'z'])
    graph = config._maybe_create_graph()
    batch_size = 250
    n = 0
    with graph.as_default():
        input = PlaceholderInput()
        input.setup(config.model.get_inputs_desc())
        with TowerContext('', is_training=False):
            config.model.build_graph(input)
        input_tensors = get_tensors_by_names(config.input_names)
        output_tensors = get_tensors_by_names(config.output_names)
        sess = config.session_creator.create_session()
        config.session_init.init(sess)
        if sess is None:
            sess = tf.get_default_session()
        start = timeit.default_timer()
        if (num % batch_size != 0):
            num_extra_img = num % batch_size
            dp = [np.random.normal(-1, 1, size=(num_extra_img, opt.Z_DIM))]
            feed = dict(zip(input_tensors, dp))
            output = sess.run(output_tensors, feed_dict=feed)
            o, zs = output[0] + 1, output[1]
            for j in range(len(o)):
                n = n + 1
                img = o[j]
                img = np.dot(img[..., :3], [0.299, 0.587, 0.114])
                img = (img - np.min(img))/np.ptp(img)
                imageio.imwrite('%s%09d.jpeg' % (sample_path,n), skimage.img_as_ubyte(img))
        for i in  range(int(num/batch_size)):
            dp = [np.random.normal(-1, 1, size=(batch_size, opt.Z_DIM))]
            feed = dict(zip(input_tensors, dp))
            output = sess.run(output_tensors, feed_dict=feed)
            o, zs = output[0] + 1, output[1]
            for j in range(len(o)):
                n = n + 1
                img = o[j]
                img = np.dot(img[..., :3], [0.299, 0.587, 0.114])
                img = (img - np.min(img))/np.ptp(img)
                imageio.imwrite('%s%09d.jpeg' % (sample_path,n), skimage.img_as_ubyte(img))
        print ("Images generated : ", str(num))
        stop = timeit.default_timer()
        print ("Time taken : ", str(stop - start))
