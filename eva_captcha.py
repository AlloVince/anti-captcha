#!/usr/bin/python
# -*- coding: utf-8 -*

from captcha.image import ImageCaptcha
import numpy as np
import random
import string


class captcha_tensorflow():
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

    def get_parameter(self):
        return self.width, self.height, self.char_num, self.characters, self.classes

    def get_batch_captcha(self, batch_size=64):
        """生成训练批量验证码"""
        # 维度：批次大小, 宽度, 高度, 通道数 (本身通道数为3，后面被转换成1)
        img_x = np.zeros([batch_size, self.height, self.width, 1])
        img_y = np.zeros([batch_size, self.char_num, self.classes])
        for i in range(batch_size):
            image = ImageCaptcha(width=self.width, height=self.height)
            captcha_str = ''.join(random.sample(self.characters, self.char_num))
            # RGB转换为灰度图
            img = image.generate_image(captcha_str).convert("L").getdata()
            img_data = np.array(img)
            img_x[i] = np.reshape(img_data, [self.height, self.width, 1]) / 255.0
            # 对验证码字符串进行二进制编码
            for j, ch in enumerate(captcha_str):
                img_y[i, j, self.characters.find(ch)] = 1
        img_y = np.reshape(img_y, (batch_size, self.char_num * self.classes))
        return img_x, img_y

    def gen_test_captcha(self):
        """生成测试验证码"""
        image = ImageCaptcha(width=self.width, height=self.height)
        captcha_str = ''.join(random.sample(self.characters, self.char_num))
        img = image.generate_image(captcha_str)
        # print img.im
        img.save(captcha_str + '.jpg')


if __name__ == '__main__':
    test = captcha_tensorflow()
    test.gen_test_captcha()
    # print test.get_batch_captcha(1)
