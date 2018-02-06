# -*- coding: UTF-8 -*-
# File: mobilenetv2.py

import argparse
import numpy as np
import os
import cv2

import tensorflow as tf

import random
from tensorpack import logger, QueueInput, InputDesc, PlaceholderInput, TowerContext
from tensorpack.models import *
from tensorpack.callbacks import *
from tensorpack.train import *
from tensorpack.dataflow import imgaug
from tensorpack.tfutils import argscope, get_model_loader
from tensorpack.tfutils.scope_utils import under_name_scope
from tensorpack.utils.gpu import get_nr_gpu

from imagenet_utils import (
    get_imagenet_dataflow,
    ImageNetModel, GoogleNetResize, eval_on_ILSVRC12)
from cfgs.config import cfg
TOTAL_BATCH_SIZE = 256


@layer_register(log_shape=True)
def DepthConv(x, out_channel, kernel_shape=3, padding='SAME', stride=1,
              W_init=None, nl=tf.identity):
    in_shape = x.get_shape().as_list()
    in_channel = in_shape[1]
    assert out_channel % in_channel == 0
    channel_mult = out_channel // in_channel
   

    if W_init is None:
        W_init = tf.variance_scaling_initializer(2.0)
    kernel_shape = [kernel_shape, kernel_shape]
    filter_shape = kernel_shape + [in_channel, channel_mult]

    W = tf.get_variable('W', filter_shape, initializer=W_init)
    conv = tf.nn.depthwise_conv2d(x, W, [1, 1, stride, stride], padding=padding, data_format='NCHW')
    # return nl(conv, name='output')
    return conv


def BN(x, i=0):
    return BatchNorm('bn{}'.format(i), x)

class Model(ImageNetModel):
    
    weight_decay = 4e-5
    def __init__(self, data_format='NCHW'):
        self.data_format = data_format

    # def _get_inputs(self):
    #     return [InputDesc(self.image_dtype, [None, cfg.h, cfg.w, 3], 'input'),
    #             InputDesc(tf.int32, [None], 'label')]

    # def _build_graph(self, inputs):
    #     image, label = inputs
    #     tf.summary.image('input_img', image)
    #     image = tf.cast(image,tf.float32)
    #     # image = (image - 127.5) / 128

    #     logits = self.get_logits(tf.transpose(image, [0, 3, 1, 2]))#NHWC TO NCHW
    #     loss = ImageNetModel.compute_loss_and_error(logits, label)


    #     if cfg.weight_decay > 0:
    #         wd_loss = regularize_cost('.*/W', tf.contrib.layers.l2_regularizer(cfg.weight_decay),
    #                                   name='l2_regularize_loss')
    #         add_moving_summary(loss, wd_loss)
    #         self.cost = tf.add_n([loss, wd_loss], name='cost')
    #     else:
    #         self.cost = tf.identity(loss, name='cost')
    #         add_moving_summary(self.cost)

    def get_logits(self, image):
        def relu6(l):
            l = PReLU('relu', l)
            return tf.maximum(l,6)
            # return l

        def bottleneck_v2(l, t, out_channels, stride=1, i=0):
            in_shape = l.get_shape().as_list()

            in_channel = in_shape[1]
            shortcut = l
            # assert in_channel==out_channel,'in_channel not equal out_channel'
            l = Conv2D('block1{}'.format(i), l, out_channel=t*in_channel)
            with tf.variable_scope('bn0{}'.format(i)): 
                l = BN(PReLU('relu0{}'.format(i), l))
            l = DepthConv('depthconv{}'.format(i), l, out_channel=t*in_channel, stride=1 if stride==1 else 2, nl=BN)
            with tf.variable_scope('bn1{}'.format(i)): 
                l = BN(PReLU('relu0{}'.format(i), l))
            l = BN(PReLU('relu1{}'.format(i), l), i)
            l = Conv2D('block2{}'.format(i), l, kernel_shape=1, out_channel= out_channels, nl=tf.identity)
            if stride==1 and out_channels==in_channel:
                return l + shortcut
                # return tf.concat([l, shortcut], 1)
            else:
                return l
        # image = tf.transpose(image, [0, 3, 1, 2])
        with argscope([Conv2D], nl=tf.identity, kernel_shape=3, stride =1, padding='SAME', data_format=self.data_format):
            l = Conv2D('covn1', image, 32, stride=2)
            # l = MaxPooling('pool1', l, 3, )
            with tf.variable_scope('bottleneck1'):
                l = bottleneck_v2(l, out_channels=16, t=1, stride=1, i='bottleneck10')

            with tf.variable_scope('bottleneck2'):
                for j in range(2):
                    with tf.variable_scope('block{}'.format(j)):
                        l = bottleneck_v2(l, out_channels=24, t=6, stride=2 if j == 0 else 1, i='bottleneck2{}'.format(j))

            with tf.variable_scope('bottleneck3'):
                for j in range(3):
                    with tf.variable_scope('block{}'.format(j)):
                        l = bottleneck_v2(l, out_channels=32, t=6, stride=2 if j == 0 else 1, i='bottleneck3{}'.format(j))
            
            with tf.variable_scope('bottleneck4'):
                for j in range(4):
                    with tf.variable_scope('block{}'.format(j)):
                        l = bottleneck_v2(l, out_channels=64, t=6, stride=2 if j == 0 else 1, i='bottleneck4{}'.format(j))
            
            with tf.variable_scope('bottleneck5'):
                for j in range(3):
                    with tf.variable_scope('block{}'.format(j)):
                        l = bottleneck_v2(l, out_channels=96, t=6, stride=1 if j == 0 else 1, i='bottleneck5{}'.format(j))
            
            with tf.variable_scope('bottleneck6'):
                for j in range(3):
                    with tf.variable_scope('block{}'.format(j)):
                        l = bottleneck_v2(l, out_channels=160, t=6, stride=2 if j== 0 else 1, i='bottleneck6{}'.format(j))
            with tf.variable_scope('bottleneck7'):
                l = bottleneck_v2(l, out_channels=320, t=6, stride=1, i='bottleneck71')
            l = Conv2D('conv2', l, 1280, kernel_shape=1)
            l = relu6(l)
            l = BN(l)
            l = AvgPooling('angpooling0', l, shape=7, data_format='NCHW')
            l = Dropout(l, cfg.dropout)
            l = Conv2D('conv3', l, cfg.class_num, kernel_shape=1)
            
            # return tf.transpose(l, [0, 2, 3, 1])
            return tf.reshape(l, [-1, 1000])

    # def _get_optimizer(self):
    #     lr = get_scalar_var('learning_rate', 0.045, summary=True)
    #     tf.summary.scalar('learning_rate-summary', lr)
    #     return tf.train.RMSPropOptimizer(lr, 0.98, use_nesterov=True)


def get_data(name, batch):
    isTrain = name == 'train'

    if isTrain:
        augmentors = [
            GoogleNetResize(crop_area_fraction=0.49),
            imgaug.RandomOrderAug(
                [imgaug.BrightnessScale((0.6, 1.4), clip=False),
                 imgaug.Contrast((0.6, 1.4), clip=False),
                 imgaug.Saturation(0.4, rgb=False),
                 # rgb-bgr conversion for the constants copied from fb.resnet.torch
                 imgaug.Lighting(0.1,
                                 eigval=np.asarray(
                                     [0.2175, 0.0188, 0.0045][::-1]) * 255.0,
                                 eigvec=np.array(
                                     [[-0.5675, 0.7192, 0.4009],
                                      [-0.5808, -0.0045, -0.8140],
                                      [-0.5836, -0.6948, 0.4203]],
                                     dtype='float32')[::-1, ::-1]
                                 )]),
            imgaug.Flip(horiz=True),
        ]
    else:
        augmentors = [
            imgaug.ResizeShortestEdge(256, cv2.INTER_CUBIC),
            imgaug.CenterCrop((cfg.h, cfg.w)),
        ]
    return get_imagenet_dataflow(
        args.data, name, batch, augmentors)


def get_config(model, nr_tower, args):
    # batch = TOTAL_BATCH_SIZE // nr_tower
    batch = args.batch_size // nr_tower

    logger.info("Running on {} towers. Batch size per tower: {}".format(nr_tower, batch))
    dataset_train = get_data('train', batch)
    dataset_val = get_data('val', batch)
    callbacks = [
        ModelSaver(),
        HyperParamSetterWithFunc('learning_rate',
                                     lambda e, x: 4.5e-2 * 0.98 ** e ),
        HumanHyperParamSetter('learning_rate'),
    ]
    infs = [ClassificationError('wrong-top1', 'val-error-top1'),
            ClassificationError('wrong-top5', 'val-error-top5')]
    if nr_tower == 1:
        # single-GPU inference with queue prefetch
        callbacks.append(InferenceRunner(QueueInput(dataset_val), infs))
    else:
        # multi-GPU inference (with mandatory queue prefetch)
        callbacks.append(DataParallelInferenceRunner(
            dataset_val, infs, list(range(nr_tower))))

    return TrainConfig(
        model=model,
        dataflow=dataset_train,
        callbacks=callbacks,
        steps_per_epoch=cfg.steps_per_epoch,
        max_epoch=cfg.max_epoch,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--data', help='ILSVRC dataset dir')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--flops', action='store_true', help='print flops and exit')
    parser.add_argument('--batch_size', default=256, type=int)
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model = Model()

    if args.eval:
        batch = args.batch_size    # something that can run on one gpu
        ds = get_data('val', batch)
        eval_on_ILSVRC12(model, get_model_loader(args.load), ds)
    elif args.flops:
        # manually build the graph with batch=1
        input_desc = [
            InputDesc(tf.float32, [1, cfg.h, cfg.w, 3], 'input'),
            InputDesc(tf.int32, [1], 'label')
        ]
        input = PlaceholderInput()
        input.setup(input_desc)
        with TowerContext('', is_training=True):
            model.build_graph(*input.get_input_tensors())

        tf.profiler.profile(
            tf.get_default_graph(),
            cmd='op',
            options=tf.profiler.ProfileOptionBuilder.float_operation())
    else:
        logger.set_logger_dir(
            os.path.join('train_log', 'mobilenetv2'))

        nr_tower = max(get_nr_gpu(), 1)
        config = get_config(model, nr_tower, args)
        if args.load:
            config.session_init = get_model_loader(args.load)
        launch_train_with_config(config, SyncMultiGPUTrainerParameterServer(nr_tower))
