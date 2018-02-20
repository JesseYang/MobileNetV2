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

def get_pred_func(args):
    sess_init = SaverRestore(args.model_path)
    model = Model()
    predict_config = PredictConfig(session_init=sess_init,
                                   model=model,
                                   input_names=["input"],
                                   output_names=["linear/output"])

    predict_func = OfflinePredictor(predict_config) 
    return predict_func

def predict(input_path, output_path, predict_func):
    ori_image = cv2.imread(input_path)
    image = cv2.resize(ori_image, (cfg.w, cfg.h))
    image = np.expand_dims(image, axis=0)

    predictions = predict_func(image)

    cls_pred = np.argmax(predictions[0])
    print(cls_pred)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='path of the model waiting for validation.', required=True)
    parser.add_argument('--input_path', help='path of the input image')
    parser.add_argument('--output_path', help='path of the output image', default='features.npy')
    args = parser.parse_args()


    predict_func = get_pred_func(args)
    predict(args.input_path, args.output_path, predict_func)
