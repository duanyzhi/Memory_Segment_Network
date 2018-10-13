from lib.config.config import FLAGS as cfg

import matplotlib.pyplot as plt
import scipy.io as si
import numpy as np
import random
import cv2
import os


class data_google:
    def __init__(self):
        self.memory_data = np.zeros((cfg.batch_size, cfg.memory_size, cfg.shot, cfg.img_size[0], cfg.img_size[1], cfg.img_size[2]))
        self.memory_label = np.zeros((cfg.batch_size, cfg.memory_size))
        self.target_data = np.zeros((cfg.batch_size, cfg.shot, cfg.img_size[0], cfg.img_size[1], cfg.img_size[2]))
        self.target_label = np.zeros(cfg.batch_size)

    def call_train(self):
        for kk in range(cfg.batch_size):
                classify_random = random.randint(0, cfg.memory_size -1)
                for ii in range(C.memory_size):
                    im_number = random.randint(1, 8734)
                    while im_number - 1 in self.train_name:
                        im_number = random.randint(1, 8734)
                    # print(im_number)
                    for jj in range(cfg.shot):
                        im_name = '{:06d}'.format(im_number) + '_' + str(jj + 1) + '.jpg'
                        # print("name", os.path.join(self.train_path, im_name))
                        self.memory_data[kk, ii, jj, ...] = cv2.resize(cv2.imread(os.path.join(self.train_path, im_name)), (cfg.img_size[1], cfg.img_size[0]))

                        if ii == classify_random:
                            target_inf = '{:06d}'.format(im_number + 1) + '_' + str(jj + 1) + '.jpg'
                            self.classify_data[kk, jj, ...] = cv2.resize(
                                cv2.imread(os.path.join(self.train_path, target_inf)),
                                (cfg.img_size[1], cfg.img_size[0]))
                    self.memory_label[kk, ii] = ii
                    if ii == classify_random:  # target slice标签
                        self.classify_label[kk] = ii

    def call_test(self):
        pass







# ------------------------------------------------------------------------------
