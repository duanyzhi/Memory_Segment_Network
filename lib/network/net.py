from novamind.ops.text_ops import list_save
from tensorflow.python.client import device_lib
from lib.datasets.read_data import data_google, data_oxford
from lib.config.config import FLAGS as cfg
from lib.layer_utils.utils import *
from .cnn import cnn_model
from .lstm import lstm
from .HMM import HMM

import tensorflow as tf
import numpy as np
import os

# multi gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # 使用 0, 1, 2, 3 一共4块GPU，编号从0开始

class msn(object):
    def __init__(self, pattern="train", data_name="google", ways="lstm"):
        self.pattern = pattern
        self.data_name = data_name
        self.ways = ways
        self.msn_out = {}

        # init model
        self.gpu = self.check_gpu()

        # load datasets
        if data_name == "google":
            self.data_msn = data_google()
            self.nns_min_loc = 20
            self.nns_max_loc = 200
        elif data_name == "oxford":
            self.data_msn = data_oxford()
            self.nns_min_loc = 20
            self.nns_max_loc = 110
        else:
            self.data_msn = data_uestc()

        self.cnn = cnn_model()
        self.lstm = lstm()
        self.hmm = HMM()

        self.placeholder()

    def placeholder(self):
        memory_shape = [None, cfg.memory_slice, cfg.Seg, cfg.img_size[0], cfg.img_size[1], cfg.img_size[2]]
        self.memory_data = tf.placeholder(tf.float32, shape= memory_shape) # reference image segment
        self.memory_label = tf.placeholder(tf.int32, shape=[None, cfg.memory_slice])
        self.memory_label_one_shot = tf.one_hot(self.memory_label, cfg.memory_slice)  # memory_label
        target_shape = [None, cfg.Seg, cfg.img_size[0], cfg.img_size[1], cfg.img_size[2]]
        self.classify_data = tf.placeholder(tf.float32, shape=target_shape)  # target image segment
        self.classify_label = tf.placeholder(tf.int32, shape=[None])
        self.classify_label_one_shot = tf.one_hot(self.classify_label, cfg.memory_slice)  # classify_label  转为one_hot
        self.is_training = tf.placeholder(tf.bool)
        self.support_feature = tf.placeholder(tf.float32, shape=[cfg.Seg, cfg.batch_size, cfg.hidden_size])

    def check_gpu(self):
        local_devices = device_lib.list_local_devices()
        gpu_names = [x.name for x in local_devices if x.device_type == 'GPU']

        cpu_names = [x.name for x in local_devices if x.device_type == 'CPU']
        gpu_num = len(gpu_names)
        cpu_num = len(cpu_names)

        print('{:1d} GPUs are detected: {}'.format(gpu_num, gpu_names))
        print('{:1d} CPUs are detected: {}'.format(cpu_num, cpu_names))
        return gpu_names

    def build_net(self):
        # CNN
        data_target = []
        data_memory = []
        support_feature = []
        for ii in range(cfg.Seg):
            target_fc, cnn_show = self.cnn(self.classify_data[:, ii, :, :, :], self.is_training)
            data_target.append(target_fc)   # output: [C.shot, batch_size, 64]
        for jj in range(cfg.memory_slice):  # memory
            data_support = []
            for kk in range(cfg.Seg):  # seg
                memory_fc, _ = self.cnn(self.memory_data[:, jj, kk, :, :, :], self.is_training)
                data_support.append(memory_fc)
            data_memory.append(data_support)

        # LSTM or fc
        if self.ways == "lstm":
            with tf.variable_scope("lstm") as lstm_name:
                data_target = tf.stack(data_target)
                data_target = tf.transpose(data_target, [1, 0, 2])
                target_feature = self.lstm(data_target)
                lstm_name.reuse_variables()
                for data in data_memory:
                    data_ = tf.stack(data)
                    data_ = tf.transpose(data_, [1, 0, 2])
                    support_feature.append(self.lstm(data_))
        else:  # fc
            with tf.device(self.gpu[1]):
                with tf.variable_scope("fc"):
                    data_target = tf.stack(data_target)
                    data_target = tf.transpose(data_target, [1, 0, 2])
                    # print("data_target", data_target)
                    fc_in_target = tf.reshape(data_target, [cfg.batch_size, cfg.Seg*cfg.hidden_size])
                    # print(fc_in_target)
                    fc_bn = tf.layers.batch_normalization(fc_in_target, name='fc_bn1_target', training=self.is_training)
                    fc_dense = tf.layers.dense(fc_bn, cfg.hidden_size, name='fc_target')
                    target_feature = tf.layers.batch_normalization(fc_dense, name='bn_fc1', training=self.is_training)
                    for index, data in enumerate(data_memory):
                        data_ = tf.stack(data)
                        data_ = tf.transpose(data_, [1, 0, 2])
                        # print("data_target2", data_)
                        fc_in = tf.reshape(data_, [cfg.batch_size, cfg.Seg*cfg.hidden_size])
                        # print(fc_in)
                        memory_bn = tf.layers.batch_normalization(fc_in, name='bn_memory1_' + str(index),
                                                                       training=self.is_training)
                        m_fc = tf.layers.dense(memory_bn, cfg.hidden_size, name='fc_memory_' + str(index))
                        # print("m_fc", m_fc)
                        memory_bn2 = tf.layers.batch_normalization(m_fc, name='bn_memory2_' + str(index),
                                                                       training=self.is_training)
                        support_feature.append(memory_bn2)

        softmax_a, similarities = cd(support_feature, target_feature)
        test_soft, self.target_sim = cd(self.support_feature, target_feature)

        self.msn_out["softmax"] = softmax_a
        self.msn_out["similarities"] = similarities
        # self.msn_out["data_target"] = data_target
        # self.msn_out["data_memory"] = data_memory
        self.msn_out["support_feature"] = support_feature
        # self.msn_out["target_feature"] = target_feature
        # self.msn_out["cnn_show"] = cnn_show

    def BP(self):
        softmax_a = self.msn_out["softmax"]

        pre_expand = tf.squeeze(tf.matmul(tf.expand_dims(softmax_a, 1), self.memory_label_one_shot))
        if len(pre_expand.shape.as_list()) == 1:
            pre_expand = tf.expand_dims(pre_expand, 0)
        top_k = tf.nn.in_top_k(pre_expand, self.classify_label, 1)  # predictions, targets targets must be 1-dimensional
        acc = tf.reduce_mean(tf.to_float(top_k))
        correct_prob = tf.reduce_sum(tf.log(tf.clip_by_value(pre_expand, 1e-10, 1.0)) * self.classify_label_one_shot, 1)
        loss = tf.reduce_mean(-correct_prob, 0)

        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        optim = tf.train.AdamOptimizer(cfg.learning_rate)
        grads = optim.compute_gradients(loss)
        train_step = optim.apply_gradients(grads)

        self.msn_out["acc"] = acc
        self.msn_out["loss"] = loss
        # self.msn_out["pre_expand"] = pre_expand
        self.msn_out["ex_ops"] = extra_update_ops
        self.msn_out["train_step"] = train_step

    def run(self):
        self.saver = tf.train.Saver()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allocator_type = 'BFC'
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)

        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.saver.restore(self.sess, cfg.oxford_ckpt)
        if self.pattern == "train":
            # self.sess.run(tf.global_variables_initializer())
            self.train()
        else:
            self.test()

    def train(self):
        msn_acc_list = []
        hmmp_acc_list = []
        hmmp_nns_acc_list = []
        loss_list = []
        for kk in range(58001, cfg.max_iter):
            memory_data, memory_label, classify_data, classify_label = self.data_msn.call_train()
            lr(kk)

            feed_dict = {self.memory_data: memory_data, self.memory_label: memory_label,
                         self.classify_data: classify_data, self.classify_label: classify_label,
                         self.is_training: True}
            with tf.device(self.gpu[0]):
                msn_show = self.sess.run(self.msn_out, feed_dict=feed_dict)
            print("iter:", kk, "loss:", msn_show["loss"])
            # print("cnn_show", msn_show["cnn_show"])
            # print("fc in target", msn_show["fc_in_target"])
            # print("cnn out", msn_show["data_target"])
            # print("data memory", msn_show["data_memory"])
            # print("target_feature", msn_show["target_feature"])
            # print("support_feature", msn_show["support_feature"])
            # print("pre_expand", msn_show["pre_expand"])
            print("softmax", msn_show["softmax"])
            print("similarities", msn_show["similarities"])
            loss_list.append(msn_show["loss"])

            # save test Acc
            if kk % 1000 == 0:
                msn_acc, hmmp_acc, hmmp_nns_acc = self.test()
                msn_acc_list.append([kk, msn_acc])
                hmmp_acc_list.append([kk, hmmp_acc])
                hmmp_nns_acc_list.append([kk, hmmp_nns_acc])

            if kk % 1000 == 0:
                self.saver.save(self.sess, 'data/ckpt/'+self.data_name+'/msn_' + str(kk) + '.ckpt')
                list_save(loss_list, "data/"+self.data_name+"/document/loss.txt")
                list_save(msn_acc_list, "data/"+self.data_name+"/document/msn_acc.txt")
                list_save(hmmp_acc_list, "data/"+self.data_name+"/document/hmmp_acc.txt")
                list_save(hmmp_nns_acc_list, "data/"+self.data_name+"/document/hmmp_nns_acc.txt")

    def test(self):
        # save memory feature as npy
        memory_iters, query_iters = self.data_msn.call_test()
        memory_npy = []
        while True:
            try:
                memory_data = next(memory_iters)
                support_feature = self.sess.run(self.msn_out["support_feature"],
                                  feed_dict={self.memory_data: memory_data,
                                             self.is_training: False})
                # memory_npy.append([s for s in support_feature])
                for nn in range(cfg.batch_size):
                    for mm in range(cfg.memory_slice):
                        o = support_feature[mm][nn]
                        memory_npy.append(o)
            except StopIteration:
                break
        memory_npy = np.array(memory_npy)  # 44, 5, 1, 512
        np.save('data/'+self.data_name+'/memory_feature.npy', memory_npy)
        print("memory_npy", len(memory_npy))

        # test query frame one by one
        loc = 0
        acc = 0
        acc_nns = 0
        baysian_information = []
        baysian_information_nns = []
        while True:
            try:
                classify_data = next(query_iters)
                sim_out = [[] for _ in range(cfg.batch_size)]
                # print("sim", sim_out)
                for kk in range(int(len(memory_npy)/cfg.memory_slice)):
                    memory_data = np.zeros([cfg.memory_slice, cfg.batch_size, cfg.hidden_size])
                    for nn in range(cfg.batch_size):
                        for mm in range(cfg.memory_slice):
                            memory_data[mm, nn, :] = memory_npy[kk*cfg.memory_slice + mm]
                    query_sim = self.sess.run(self.target_sim,
                                 feed_dict={self.classify_data: classify_data,
                                            self.support_feature: memory_data,
                                            self.is_training: False})
                    # print("query_sim", len(query_sim), query_sim)
                    for nn in range(cfg.batch_size):
                        sim_out[nn].extend(query_sim[nn])

                for nn in range(cfg.batch_size):
                    # wo NNS
                    # baysian
                    arg_sim = arg_sort(sim_out[nn], cfg.maxN, True)
                    # print("arg_sim", loc, arg_sim)
                    baysian_information.append(arg_sim)

                    hmm_loc = arg_sim[0][0]
                    if abs(loc - hmm_loc) <= cfg.loc_threshold:
                        acc += 1
                    print("HMM loc:", hmm_loc, "Loc label:", loc)
                    # NNS Location
                    if self.nns_min_loc < loc < self.nns_max_loc:
                        nns_data = sim_out[nn][loc - self.nns_min_loc:loc + self.nns_min_loc]
                        location_index = nns_data.index(max(nns_data))
                        NNS_location_index = location_index + loc - self.nns_min_loc
                        # print("real label NNS:", loc, "query label NNS:", NNS_location_index)
                        if abs(loc - NNS_location_index) <= cfg.loc_threshold:
                            acc_nns += 1

                        # baysian
                        arg_sim_nns = arg_sort(nns_data, cfg.maxN, True)
                        arg_sim_nns = [[a[0] + loc - self.nns_min_loc, a[1]] for a in arg_sim_nns]
                        # print("arg_sim_nns", loc, arg_sim_nns)
                        baysian_information_nns.append(arg_sim_nns)
                    else:
                        arg_sim_nns = arg_sort(sim_out[nn], cfg.maxN, True)
                        # print("arg_sim", loc, arg_sim_nns)
                        baysian_information_nns.append(arg_sim_nns)
                        nns_loc = arg_sim_nns[0][0]
                        if abs(loc - nns_loc) <= cfg.loc_threshold:
                            acc_nns += 1

                    loc += 1
            except StopIteration:
                break
        list_save(baysian_information, "data/"+self.data_name+"/document/msn.txt")
        list_save(baysian_information_nns, "data/"+self.data_name+"/document/msn_nns.txt")
        hmmp_acc = self.hmm(baysian_information)
        hmmp_nns_acc = self.hmm(baysian_information_nns)
        print("Memory Segment Network Test Accaury:", acc / loc)
        print("Memory Segment Network with NNS Test Accaury:", acc_nns / loc)
        return acc / loc, hmmp_acc, hmmp_nns_acc
