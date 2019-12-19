import tensorflow as tf
from tensorpack import ModelDescBase
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.utils.argtools import memoized


class GANModelDesc(ModelDescBase):
    def collect_variables(self, g_scope='gen', d_scope='discrim'):
        self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, g_scope)
        assert self.g_vars
        self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, d_scope)
        assert self.d_vars

    def build_losses(self, logits_real, logits_fake):
        with tf.name_scope("GAN_loss"):
            score_real = tf.sigmoid(logits_real)
            score_fake = tf.sigmoid(logits_fake)
            tf.summary.histogram('score-real', score_real)
            tf.summary.histogram('score-fake', score_fake)

            with tf.name_scope("discrim"):
                d_loss_pos = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=logits_real, labels=tf.ones_like(logits_real)), name='loss_real')
                d_loss_neg = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=logits_fake, labels=tf.zeros_like(logits_fake)), name='loss_fake')

                d_pos_acc = tf.reduce_mean(tf.cast(score_real > 0.5, tf.float32), name='accuracy_real')
                d_neg_acc = tf.reduce_mean(tf.cast(score_fake < 0.5, tf.float32), name='accuracy_fake')

                d_accuracy = tf.add(.5 * d_pos_acc, .5 * d_neg_acc, name='accuracy')
                self.d_loss = tf.add(.5 * d_loss_pos, .5 * d_loss_neg, name='loss')

            with tf.name_scope("gen"):
                self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=logits_fake, labels=tf.ones_like(logits_fake)), name='loss')
                g_accuracy = tf.reduce_mean(tf.cast(score_fake > 0.5, tf.float32), name='accuracy')

            add_moving_summary(self.g_loss, self.d_loss, d_accuracy, g_accuracy)

    @memoized
    def get_optimizer(self):
        return self._get_optimizer()