# utils
import tensorflow as tf
from lib.config.config import FLAGS as cfg

#  cosine distance
def cd(memory_images, x_target):
    cosine_distance = []
    """
    memory_images: [5, batch size, 512]
    x_target: [batch_size, feature]
    """
    for support_image in tf.unstack(memory_images, axis=0):
        sum_support = tf.reduce_sum(tf.square(support_image), 1, keep_dims=True)  # 先平方，再按行求和（一个batch求和）
        support_magnitude = tf.rsqrt(tf.clip_by_value(sum_support, 1e-10, float("inf")))  # 将值变到固定大小
        k1 = tf.expand_dims(x_target, 1)  # [16, 1, 64]
        k2 = tf.expand_dims(support_image, 2)  # [16, 64,1]
        dot_product = tf.matmul(k1, k2)
        dot_product = tf.squeeze(dot_product, [1, ])  # [32, 1]
        cosine_similarity = dot_product * support_magnitude  #
        cosine_distance.append(cosine_similarity)
    similarities = tf.concat(axis=1, values=cosine_distance)           # [batch_size, 16]
    softmax_a = tf.nn.softmax(similarities)
    return softmax_a, similarities

def lr(kk):    # 100000
    if kk < 58101:
        cfg.learning_rate = 0.001
    # elif 1000 < kk < 20001:
    #     cfg.learning_rate = 0.001
    # elif 20000 < kk < 30001:
    #     cfg.learning_rate = 0.0005
    # elif 30000 < kk < 40000:
    #     cfg.learning_rate = 0.0001
    else:
        cfg.learning_rate = 0.0001

def arg_sort(raw, n, flags=False):
    '''
    @raw 一维列表
    @n 要返回n个最大值索引
    @flags 默认False求最小值 False返回索引最大值
    根据列表返回列表的前n个最大值的索引位置
    '''
    copy_raw = raw[::]
    copy_raw = [[index, node]for index, node in enumerate(copy_raw)]
    copy_raw.sort(key=lambda f:f[1], reverse=flags)
    return [num for num in copy_raw[:n]]
