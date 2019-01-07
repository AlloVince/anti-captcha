import json
import falcon
from falcon_multipart.middleware import MultipartMiddleware
from train import crack_captcha_cnn, vec2text, X, keep_prob, model_path
from gen_captcha import MAX_CAPTCHA, CHAR_SET_LEN
import numpy as np
import tensorflow as tf
import os
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

output = crack_captcha_cnn()
sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint(model_path))



def load_image(image_body):
    nparr = np.fromstring(image_body, np.uint8)
    original_image = cv2.imdecode(nparr, 1)
    image = cv2.medianBlur(original_image, 5)
    ret, th1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    th1 = cv2.cvtColor(th1, cv2.COLOR_BGR2GRAY)
    height, width = th1.shape
    revert = th1.copy()
    count = 0
    for i in range(height):
        for j in range(width):
            if th1[i, j] == 0:
                count += 1
            revert[i, j] = (255 - th1[i, j])

    return revert if count > height * width / 2 else th1


def crack_captcha(captcha_image):
    predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    text_list = sess.run(predict, feed_dict={X: [captcha_image], keep_prob: 1})
    text = text_list[0].tolist()
    vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
    i = 0
    for n in text:
        vector[i * CHAR_SET_LEN + n] = 1
        i += 1
    return vec2text(vector)


class Resource(object):

    def on_get(self, req, resp):
        index = ''
        with open('index.html', 'r') as f:
            index = f.read()
        resp.body = index
        resp.content_type = 'text/html'
        resp.status = falcon.HTTP_200

    def on_post(self, req, resp):
        image = req.get_param('image')
        image_matrix = load_image(image.file.read())
        predict_text = crack_captcha(image_matrix.flatten() / 255)
        doc = {
            'predict': predict_text,
        }
        resp.body = json.dumps(doc, ensure_ascii=False)
        resp.status = falcon.HTTP_200


api = application = falcon.API(middleware=[MultipartMiddleware()])
api.add_route('/', Resource())
