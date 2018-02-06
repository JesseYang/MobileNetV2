from easydict import EasyDict as edict

cfg = edict()

cfg.h = 224
cfg.w = 224

cfg.class_num = 1000

#dropout need to keep
cfg.dropout = 0.5

cfg.steps_per_epoch = 5000
cfg.max_epoch = 160
cfg.weight_decay = 4e-5
