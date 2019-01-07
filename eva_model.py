# -*-coding:utf-8-*-

import tensorflow as tf
import string


class create_model():
    """模型构建"""

    def __init__(self, width=160, height=60, char_num=4,
                 characters=string.digits + string.ascii_uppercase + string.ascii_lowercase):
        # 验证码的宽度
        self.width = width
        # 验证码的高度
        self.height = height
        # 验证码字符数量
        self.char_num = char_num
        # 验证码字符集
        self.characters = characters
        # 验证码字符集种类
        self.classes = len(characters)

    def weight_init(self, shape, name):
        """获得权重参数"""
        return tf.get_variable(name, shape, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))

    def bias_init(self, shape, name):
        """获得偏置参数"""
        return tf.get_variable(name, shape, initializer=tf.constant_initializer(0.0))

    def conv2d(self, x, conv_w):
        """计算二维数据的卷积，两个维度的步长都是1，填充边缘"""
        return tf.nn.conv2d(x, conv_w, strides=[1, 1, 1, 1], padding="SAME")

    def max_pool(self, x, size):
        """计算池化，池化区域为size*size，填充边缘"""
        return tf.nn.max_pool(x, ksize=[1, size, size, 1], strides=[1, size, size, 1], padding="SAME")

    def inference(self, input_data):
        """定义网络结构，进行前向计算"""
        # 卷积层
        with tf.name_scope("conv1"):
            # 卷积核大小是5x5，输入通道为1，输出通道为32
            w_conv1 = self.weight_init([5, 5, 1, 32], "w_conv1")
            b_conv1 = self.bias_init([32], "b_conv1")
            # 卷积之后，图片大小变成160x60 (160/1=160, 60/1=60)
            h_conv1 = tf.nn.relu(self.conv2d(input_data, w_conv1) + b_conv1)
            # 池化之后，图片大小变成80x30 (160/2=80, 60/2=30)
            h_pool1 = self.max_pool(h_conv1, 2)
        with tf.name_scope("conv2"):
            # 卷积核大小是5x5，输入通道为32，输出通道为64
            w_conv2 = self.weight_init([5, 5, 32, 64], "w_conv2")
            b_conv2 = self.bias_init([64], "b_conv2")
            # 卷积之后，图片大小变为80x30 (80/1=66，30/1=30)
            h_conv2 = tf.nn.relu(self.conv2d(h_pool1, w_conv2) + b_conv2)
            # 池化之后，图片大小变为40x15 (80/2=40, 30/2=15)
            h_pool2 = self.max_pool(h_conv2, 2)
        with tf.name_scope("conv3"):
            # 卷积核大小为5x5，输入通道为64，输出通道为64
            w_conv3 = self.weight_init([5, 5, 64, 64], "w_conv3")
            b_conv3 = self.bias_init([64], "b_conv3")
            # 卷积之后，图片大小变为40x15 (40/1=40, 15/1=15)
            h_conv3 = tf.nn.relu(self.conv2d(h_pool2, w_conv3) + b_conv3)
            # 池化之后，图片大小变为20x8 (40/2=20, 15/2=8)
            h_pool3 = self.max_pool(h_conv3, 2)
        # 全连接层
        with tf.name_scope("fc1"):
            # 将池化后的数据拉长为20*8*64=10240的一维向量
            # 再做全连接，第一层输入为10240，输出为1024
            w_fc1 = self.weight_init([20 * 8 * 64, 1024], "w_fc1")
            b_fc1 = self.bias_init([1024], "b_fc1")
            h_fc1 = tf.nn.relu(tf.matmul(tf.reshape(h_pool3, [-1, 20 * 8 * 64]), w_fc1) + b_fc1)
        with tf.name_scope("fc2"):
            # 第二层输入长度为1024，输出长度为验证码的种类数
            w_fc2 = self.weight_init([1024, self.char_num * self.classes], "w_fc2")
            b_fc2 = self.bias_init([self.char_num * self.classes], "b_fc2")
            h_fc2 = tf.matmul(h_fc1, w_fc2) + b_fc2
        return h_fc2
