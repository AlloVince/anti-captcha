# coding:utf-8
import asyncio
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
import requests

matplotlib.use('TkAgg')

# 验证码中的字符, 就不用汉字了
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']
# ALPHABET = []
char_set = number + alphabet + ALPHABET + ['_']  # 如果验证码长度小于4, '_'用来补齐
# IMAGE_WIDTH = 160
# IMAGE_HEIGHT = 60
# MAX_CAPTCHA = 4
IMAGE_WIDTH = 200
IMAGE_HEIGHT = 80
MAX_CAPTCHA = 5
CHAR_SET_LEN = len(char_set)


# image = np.array(Image.open(BytesIO(response.content)))

async def gen_captcha_text_and_image():
    response = requests.get('http://localhost:8888/captcha.php')
    captcha_text = response.headers['X-Phrase']

    # open('samples/' + captcha_text + '.jpg', 'wb').write(response.content)
    nparr = np.fromstring(response.content, np.uint8)
    original_image = cv2.imdecode(nparr, 1)
    image = cv2.medianBlur(original_image, 5)
    ret, th1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    return captcha_text, th1, original_image


async def main():
    text, image, original_image = await gen_captcha_text_and_image()
    f = plt.figure()
    plt.subplot(2, 2, 1), plt.imshow(original_image)
    plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 2), plt.imshow(image)
    plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()
