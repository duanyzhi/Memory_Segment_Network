# coding=utf-8

from lib.config.config import FLAGS as cfg

import matplotlib.pyplot as plt
import scipy.io as si
import numpy as np
import random
import glob
import cv2
import os


class data_google(object):
    def __init__(self):
        # datasets numpy
        self.memory_data = np.zeros((cfg.batch_size, cfg.memory_slice, cfg.Seg, cfg.img_size[0], cfg.img_size[1], cfg.img_size[2]))
        self.memory_label = np.zeros((cfg.batch_size, cfg.memory_slice))
        self.target_data = np.zeros((cfg.batch_size, cfg.Seg, cfg.img_size[0], cfg.img_size[1], cfg.img_size[2]))
        self.target_label = np.zeros(cfg.batch_size)
        # test_img = glob.glob('data/Google/test/*.jpg')

        self.index = {"train":0, "test":0}
        self.discreate_train_location = [0, 139, 304, 346, 405, 627, 735, 768,
                         769, 770, 771, 776, 874, 879, 898, 908, 921, 928, 942,
                         957, 958, 966, 1065, 1171, 1223, 1224, 1276, 1333, 1334,
                         1354, 1479, 3078, 4402, 8259, 8538, 8636, 8735]   # 因为数据集不完全是连续的
        self.discreate_test_location = [9976]
        self.test_query, self.test_memory = self.get_google_test()

    # get test img
    @staticmethod
    def get_google_test():
        img_list = os.listdir(cfg.Google_test_data)
        img_list.sort()

        query_list = []
        memory_list = []
        for n in range(int(len(img_list)/6)):
            if n % 2 == 0:
                query_list.append(img_list[n*6 + 1 :n*6 + 6])
            else:
                memory_list.append(img_list[n*6 + 1:n*6 + 6])
        return query_list, memory_list

    def call_train(self):
        for kk in range(cfg.batch_size):
            target_num = random.randint(0, cfg.memory_slice -1)
            for ii in range(cfg.memory_slice):
                im_number = random.randint(1, 8734)
                while im_number - 1 in self.discreate_train_location:
                    im_number = random.randint(1, 8734)
                for jj in range(cfg.Seg):   # one location: .._1, ..., ..._5
                    im_name = '{:06d}'.format(im_number) + '_' + str(jj + 1) + '.jpg'
                    # print("memory", os.path.join(cfg.Google_train_data, im_name))
                    self.memory_data[kk, ii, jj, ...] = cv2.resize(cv2.imread(os.path.join(cfg.Google_train_data, im_name)), (cfg.img_size[1], cfg.img_size[0]))

                    if ii == target_num:
                        target_inf = '{:06d}'.format(im_number + 1) + '_' + str(jj + 1) + '.jpg'
                        # print("target", os.path.join(cfg.Google_train_data, target_inf))
                        self.target_data[kk, jj, ...] = cv2.resize(cv2.imread(os.path.join(cfg.Google_train_data, target_inf)), (cfg.img_size[1], cfg.img_size[0]))
                self.memory_label[kk, ii] = ii
                if ii == target_num:  # target slice标签
                    self.target_label[kk] = ii
        return (self.memory_data - cfg.Google_mean) / cfg.Google_std, self.memory_label, \
                   (self.target_data - cfg.Google_mean) / cfg.Google_std, self.target_label

    def call_test(self):
        memory_iter = []
        for kk in range(int(len(self.test_memory)/cfg.memory_slice)):  # 5 memory_slice for once model
            # print(self.test_memory[kk*cfg.memory_slice:(kk+1)*cfg.memory_slice])
            for ii, mem_im in enumerate(self.test_memory[kk*cfg.memory_slice:(kk+1)*cfg.memory_slice]):
                # print(mem_im)
                for jj, one_im in enumerate(mem_im):
                    im_name = os.path.join(cfg.Google_test_data, one_im)
                    # print("im_name", im_name)
                    self.memory_data[0, ii, jj, ...] = cv2.resize(cv2.imread(im_name), (cfg.img_size[1], cfg.img_size[0]))
            memory_iter.append((self.memory_data - cfg.Google_mean) / cfg.Google_std)

        target_iter = []
        for kk in range(len(self.test_query)):
            for jj, one_im in enumerate(self.test_query[kk]):
                im_name = os.path.join(cfg.Google_test_data, one_im)
                # print("im_name", im_name)
                self.target_data[0, jj, ...] = cv2.resize(cv2.imread(im_name), (cfg.img_size[1], cfg.img_size[0]))
            target_iter.append((self.target_data - cfg.Google_mean) / cfg.Google_std)
        return iter(memory_iter), iter(target_iter)

class data_oxford(object):
    def __init__(self):
        # datasets numpy
        self.memory_data = np.zeros((cfg.batch_size, cfg.memory_slice, cfg.Seg, cfg.img_size[0], cfg.img_size[1], cfg.img_size[2]))
        self.memory_label = np.zeros((cfg.batch_size, cfg.memory_slice))
        self.target_data = np.zeros((cfg.batch_size, cfg.Seg, cfg.img_size[0], cfg.img_size[1], cfg.img_size[2]))
        self.target_label = np.zeros(cfg.batch_size)

        memory_list = sorted(os.listdir(cfg.Oxford_train_data))
        target_list = sorted(os.listdir(cfg.Oxford_test_data))

        self.train_memory = memory_list[:5380]  # 前5000张训练
        self.train_target = target_list[:5380]  # 后980张测试

        self.test_memory = memory_list[5380:]
        self.test_query = target_list[5380:]

        self.count = 0
        self.begin = 0

        self.len_M = cfg.memory_slice*cfg.Seg

    def call_train(self):
        if (self.begin + (self.count+1)*cfg.batch_size*self.len_M) > len(self.train_memory):
            self.count = 0
            self.begin = np.random.randint(0, 25)

        begin_token = self.begin + self.count*cfg.batch_size*self.len_M

        for kk in range(cfg.batch_size):
            target_num = random.randint(0, cfg.memory_slice -1)
            for ii in range(cfg.memory_slice):
                for jj in range(cfg.Seg):
                    im_name = self.train_memory[begin_token + kk*self.len_M + ii*cfg.memory_slice + jj]
                    self.memory_data[kk, ii, jj, ...] = cv2.resize(cv2.imread(os.path.join(cfg.Oxford_train_data, im_name)), (cfg.img_size[1], cfg.img_size[0]))

                    if ii == target_num:
                        target_inf = self.train_target[begin_token + kk*self.len_M + ii*cfg.memory_slice + jj]
                        self.target_data[kk, jj, ...] = cv2.resize(cv2.imread(os.path.join(cfg.Oxford_test_data, target_inf)), (cfg.img_size[1], cfg.img_size[0]))
                self.memory_label[kk, ii] = ii
                if ii == target_num:  # target slice标签
                    self.target_label[kk] = ii
        self.count += 1
        return (self.memory_data - cfg.oxford_memory_mean) / cfg.oxford_memory_std, self.memory_label, \
                   (self.target_data - cfg.oxford_memory_mean) / cfg.oxford_memory_std, self.target_label


    def call_test(self):
        memory_iter = []
        for kk in range(int(len(self.test_memory)/self.len_M/cfg.batch_size)):  # 5*batch memory_slice for once model
            # print(self.test_memory[kk*cfg.memory_slice:(kk+1)*cfg.memory_slice])
            memory_batch_data = self.test_memory[
            kk * (self.len_M * cfg.batch_size):(kk + 1) * (self.len_M * cfg.batch_size)]
            # print("memory_batch_data", len(memory_batch_data))
            # print(memory_batch_data)
            for nn in range(cfg.batch_size):
                batch_data = memory_batch_data[nn*self.len_M:(nn+1)*self.len_M]
                # print("batch_data", batch_data)
                for mm, mem_im in enumerate(batch_data):
                    im_name = os.path.join(cfg.Oxford_train_data, mem_im)
                    # print("im_name1", im_name)
                    ii, jj = mm//cfg.Seg, mm%cfg.Seg
                    # print(mm, ii, jj)
                    self.memory_data[nn, ii, jj, ...] = cv2.resize(cv2.imread(im_name), (cfg.img_size[1], cfg.img_size[0]))
            memory_iter.append((self.memory_data - cfg.oxford_target_mean) / cfg.oxford_target_std)

        target_iter = []
        for kk in range(int(len(self.test_query)/cfg.batch_size/cfg.Seg)):
            test_batch_data = self.test_query[kk*(cfg.batch_size*cfg.Seg):(kk+1)*(cfg.batch_size*cfg.Seg)]
            print("test_batch_data", test_batch_data)
            for nn in range(cfg.batch_size):
                for jj, one_im in enumerate(test_batch_data[nn*cfg.Seg:(nn+1)*cfg.Seg]):
                    im_name = os.path.join(cfg.Oxford_test_data, one_im)
                    # print("im_name2", im_name)
                    self.target_data[nn, jj, ...] = cv2.resize(cv2.imread(im_name), (cfg.img_size[1], cfg.img_size[0]))
            target_iter.append((self.target_data - cfg.oxford_target_mean) / cfg.oxford_target_std)
        return iter(memory_iter), iter(target_iter)







# ------------------------------------------------------------------------------
