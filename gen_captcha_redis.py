# coding:utf-8
import redis
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
import requests
import platform

np.set_printoptions(threshold=np.nan)

if platform.system() == 'Darwin':
    matplotlib.use('TkAgg')

# 验证码中的字符, 就不用汉字了
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']
# ALPHABET = []
char_set = number + alphabet + ALPHABET + ['_']  # 如果验证码长度小于4, '_'用来补齐
IMAGE_WIDTH = 200
IMAGE_HEIGHT = 80
MAX_CAPTCHA = 5
CHAR_SET_LEN = len(char_set)


def pr(matrix, sim=False):
    if sim:
        s = [[str(1 if e == 255 else 0) for e in row] for row in matrix]
    else:
        s = [[str(e) for e in row] for row in matrix]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = ''.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print('\n'.join(table))


def gen_captcha_text_and_image(image, captcha_text):
    nparr = np.fromstring(image, np.uint8)
    original_image = cv2.imdecode(nparr, 1)
    image = cv2.medianBlur(original_image, 5)
    ret, th1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    th1 = cv2.cvtColor(th1, cv2.COLOR_BGR2GRAY)
    # return captcha_text, th1, original_image
    #
    height, width = th1.shape
    revert = th1.copy()
    count = 0
    for i in range(height):
        for j in range(width):
            if th1[i, j] == 0:
                count += 1
            revert[i, j] = (255 - th1[i, j])

    return captcha_text, revert if count > height * width / 2 else th1, original_image


def main():
    redis_client = redis.Redis(connection_pool=redis.ConnectionPool.from_url('unix:///tmp/redis.sock'))
    img = redis_client.get('captcha:2hqyx')
    text, image, original_image = gen_captcha_text_and_image(img, 'abcde')
    # pr(image)
    a = image.flatten() / 255
    # print(a)
    plt.figure()
    plt.subplot(2, 2, 1), plt.imshow(original_image)
    plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 2), plt.imshow(image, 'gray')
    plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == '__main__':

    main()