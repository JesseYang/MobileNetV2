#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import uuid
import shutil
import ntpath
import numpy as np
from scipy import misc
import argparse
import json
import cv2

from tensorpack.tfutils.sesscreate import SessionCreatorAdapter, NewSessionCreator
from tensorflow.python import debug as tf_debug

from tensorpack import *

try:
    from .cfgs.config import cfg
except Exception:
    from cfgs.config import cfg


try:
    from .mobilenetv2 import Model
except Exception:
    from mobilenetv2 import Model

feat_names = ["network_input",
              "covn1/output",
              "bottleneck1/conv2/bn/output",
              "bottleneck2/block0/conv2/bn/output",
              "bottleneck2/block1/add",
              "bottleneck3/block0/conv2/bn/output",
              "bottleneck3/block1/add",
              "bottleneck3/block2/add",
              "bottleneck4/block0/conv2/bn/output",
              "bottleneck4/block1/add",
              "bottleneck4/block2/add",
              "bottleneck4/block3/add",
              "bottleneck5/block0/conv2/bn/output",
              "bottleneck5/block1/add",
              "bottleneck5/block2/add",
              "bottleneck6/block0/conv2/bn/output",
              "bottleneck6/block1/add",
              "bottleneck6/block2/add",
              "bottleneck7/conv2/bn/output",
              "conv2/output",
              "linear/output"]

def get_pred_func(args):
    sess_init = SaverRestore(args.model_path)
    model = Model()
    predict_config = PredictConfig(session_init=sess_init,
                                   model=model,
                                   input_names=["input"],
                                   output_names=feat_names)

    predict_func = OfflinePredictor(predict_config) 
    return predict_func

def do_export(input_path, output_path, predict_func):
    ori_image = cv2.imread(input_path)
    cvt_clr_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(cvt_clr_image, (cfg.w, cfg.h))
    image = np.expand_dims(image, axis=0)

    predictions = predict_func(image)

    feat_dict = { }
    for feat_idx, feat_name in enumerate(feat_names):
        # key_name = feat_name.split('/')[0]
        key_name = feat_name.replace('/', '_')
        feat_dict[key_name] = predictions[feat_idx][0]

    np.save(output_path, feat_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='path of the model waiting for validation.', required=True)
    parser.add_argument('--input_path', help='path of the input image')
    parser.add_argument('--output_path', help='path of the output image', default='features.npy')
    args = parser.parse_args()


    predict_func = get_pred_func(args)
    do_export(args.input_path, args.output_path, predict_func)
