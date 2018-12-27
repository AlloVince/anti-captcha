import asyncio

from train import crack_captcha_cnn, convert2gray, vec2text, X, keep_prob, model_path
from gen_captcha import gen_captcha_text_and_image, MAX_CAPTCHA, CHAR_SET_LEN
import numpy as np
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def crack_captcha(captcha_image):
    output = crack_captcha_cnn()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(model_path))

        predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
        text_list = sess.run(predict, feed_dict={X: [captcha_image], keep_prob: 1})
        text = text_list[0].tolist()
        vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
        i = 0
        for n in text:
            vector[i * CHAR_SET_LEN + n] = 1
            i += 1
        return vec2text(vector)


def main():
    text, image, o = gen_captcha_text_and_image()
    image = image.flatten() / 255
    predict_text = crack_captcha(image)
    print("匹配: {} 正确: {}  预测: {}".format(text == predict_text, text, predict_text))


if __name__ == '__main__':
    main()
