import cv2
from math import pow, exp
from tf_pixelcnn.models import PixelCNN
from tf_pixelcnn.layers import *
import logging


class DotDict(object):
    def __init__(self, dict):
        self.dict = dict

    def __getattr__(self, name):
        return self.dict[name]

    def update(self, name, val):
        self.dict[name] = val

    # can delete this later
    def get(self, name):
        return self.dict[name]


# specify command line arguments using flags
FLAGS = DotDict({
    'img_height': 42,
    'img_width': 42,
    'channel': 1,
    'data': 'mnist',
    'conditional': False,
    'num_classes': None,
    'filter_size': 3,
    'init_fs': 7,
    'f_map': 16,
    'f_map_fc': 16,
    'colors': 8,
    'parallel_workers': 1,
    'layers': 3,
    'epochs': 25,
    'batch_size': 16,
    'model': '',
    'data_path': 'data',
    'ckpt_path': 'ckpts',
    'samples_path': 'samples',
    'summary_path': 'logs',
    'restore': True,
})


class PixelBonus(object):
    """
    Tensorflow wrapper for PixelCNN model and exploration bonus; borrowed implementation
    """
    def __init__(self, FLAGS):
        # init model
        self.X = tf.placeholder(
            tf.float32,
            shape=[None, FLAGS.img_height, FLAGS.img_width, FLAGS.channel])
        self.model = PixelCNN(self.X, FLAGS)
        self.flags = FLAGS
        # init optimizer
        self.optimizer = tf.train.RMSPropOptimizer(
            learning_rate=1e-3,decay=0.95,momentum=0.9).minimize(self.model.loss)

        # make sure GPU doesn't use all of the available memory
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        self.sess.run(tf.global_variables_initializer())
        self.frame_shape = (FLAGS.img_height, FLAGS.img_width)
        # self.max_val = np.finfo(np.float32).max - 1e-10

    def bonus(self, obs, t):
        """
        Calculate exploration bonus with observed frame
        :param obs:
        :return:
        """
        # reshape image so that it's 42 x 42
        frame = cv2.resize(obs, self.frame_shape)

        # [0,1] pixel values
        frame = (np.reshape(frame, [42, 42, 1]) / 255.)
        frame = np.expand_dims(frame, 0)  # (N, Y, X, C)

        # compute PG
        log_prob = self.density_model_logprobs(frame, update=True)

        # train a single additional step with the same observation; no update
        log_recoding_prob = self.density_model_logprobs(frame, update=False)

        # compute prediction gain
        pred_gain = max(0, log_recoding_prob - log_prob)

        # save log loss
        # nll = self.sess.run(self.model.nll, feed_dict={self.X: frame})

        # calculate intrinsic reward
        intrinsic_reward = pow((exp(0.1*pow(t + 1, -0.5) * pred_gain) - 1), 0.5)

        return intrinsic_reward

    def density_model_logprobs(self, img, update=False):
        """
        compute log loss WITHOUT updating parameters
        :param img:
        :return:
        """
        if update:
            _, logprob, target_idx = self.sess.run([
                self.optimizer, self.model.log_probs, self.model.target_idx], feed_dict={
                self.X: img})
        else:
            logprob, target_idx = self.sess.run([
                self.model.log_probs, self.model.target_idx], feed_dict={self.X: img})

        pred_prob = logprob[np.arange(FLAGS.img_height * FLAGS.img_width),
                            target_idx].sum()

        return pred_prob

    # def sum_pixel_log_probs(self, img, log_prob_func):
    #     """
    #     compute log loss over each individual pixel
    #     :param img:
    #     :param log_prob_func:
    #     :return:
    #     """
    #     total_log_prob = 0.
    #
    #     for y in range(img.shape[0]):
    #         for x in range(img.shape[1]):
    #             total_log_prob += log_prob_func(img[y,x])
    #     return total_log_prob
