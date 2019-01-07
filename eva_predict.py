# -*-coding:utf-8-*-
from PIL import Image, ImageFilter
import tensorflow as tf
import numpy as np
import string
import sys
import eva_captcha
import eva_model


def main():
    # 获取参数
    captcha_gen = eva_captcha.captcha_tensorflow()
    width, height, char_num, characters, classes = captcha_gen.get_parameter()
    # 读取图片
    gray_image = Image.open(sys.argv[1]).convert('L')
    img = np.array(gray_image.getdata())
    test_x = np.reshape(img, [height, width, 1]) / 255.0
    # 定义占位符
    img_x = tf.placeholder(tf.float32, [None, height, width, 1])
    # 预测
    captcha_mod = eva_model.create_model()
    pre_y = captcha_mod.inference(img_x)
    predict = tf.argmax(tf.reshape(pre_y, [-1, char_num, classes]), 2)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # 加载模型
        saver.restore(sess, "model/00/captcha_model.ckpt")
        pre_list = sess.run(predict, feed_dict={img_x: [test_x]})
        for i in pre_list:
            s = ''
            for j in i:
                s += characters[j]
            print(s)


if __name__ == '__main__':
    main()
