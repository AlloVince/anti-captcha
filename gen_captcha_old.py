# coding:utf-8
import asyncio

from captcha.image import ImageCaptcha, random_color  # pip install captcha
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import random, time, os, math
from PIL.ImageDraw import Draw
import cv2

matplotlib.use('TkAgg')

# 验证码中的字符, 就不用汉字了
# number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
# alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
#             'v', 'w', 'x', 'y', 'z']
# ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
#             'V', 'W', 'X', 'Y', 'Z']
number = ['0', '1']
alphabet = []
ALPHABET = []
char_set = number + alphabet + ALPHABET + ['_']  # 如果验证码长度小于4, '_'用来补齐
# IMAGE_WIDTH = 160
# IMAGE_HEIGHT = 60
# MAX_CAPTCHA = 4
IMAGE_WIDTH = 200
IMAGE_HEIGHT = 80
MAX_CAPTCHA = 5
CHAR_SET_LEN = len(char_set)

table = []
for i in range(256):
    table.append(i * 1.97)


class MyImageCaptcha(ImageCaptcha):
    def create_background(self, background):
        image = Image.new('RGB', (self._width, self._height), background)
        return image

    def draw_wave(self, image, start, end, frequency, amp, width, color):
        draw = Draw(image)
        sx, sy = start
        ex, ey = end
        x_len = abs(ex - sx)
        y_len = abs(ey - sy)
        points = []
        for i in range(0, x_len):
            x = sx + i if ex > sx else sx - i
            y_diff = i * y_len / x_len
            y = sy + y_diff if ey > sy else sy - y_diff
            y += math.sin(math.radians(x) * frequency) * amp
            for w in range(1, width):
                points.append((x, y + w))
            points.append((x, y))

        draw.point(points, fill=color)
        return image

    def draw_text(self, image, chars, color):
        """Create the CAPTCHA image itself.

        :param chars: text to be generated.
        :param color: color of the text.
        :param background: color of the background.

        The color should be a tuple of 3 numbers, such as (0, 255, 255).
        """
        draw = Draw(image)

        def _draw_character(c):
            font = random.choice(self.truefonts)
            w, h = draw.textsize(c, font=font)

            dx = random.randint(0, 4)
            dy = random.randint(0, 6)
            im = Image.new('RGBA', (w + dx, h + dy))
            Draw(im).text((dx, dy), c, font=font, fill=color)

            # rotate
            im = im.crop(im.getbbox())
            im = im.rotate(random.uniform(-30, 30), Image.BILINEAR, expand=1)

            # warp
            dx = w * random.uniform(0.1, 0.3)
            dy = h * random.uniform(0.2, 0.3)
            x1 = int(random.uniform(-dx, dx))
            y1 = int(random.uniform(-dy, dy))
            x2 = int(random.uniform(-dx, dx))
            y2 = int(random.uniform(-dy, dy))
            w2 = w + abs(x1) + abs(x2)
            h2 = h + abs(y1) + abs(y2)
            data = (
                x1, y1,
                -x1, h2 - y2,
                w2 + x2, h2 + y2,
                w2 - x2, -y1,
            )
            im = im.resize((w2, h2))
            im = im.transform((w, h), Image.QUAD, data)
            return im

        images = []
        for c in chars:
            if random.random() > 0.5:
                images.append(_draw_character(" "))
            images.append(_draw_character(c))

        text_width = sum([im.size[0] for im in images])

        width = max(text_width, self._width)
        image = image.resize((width, self._height))

        average = int(text_width / len(chars))
        rand = int(0.25 * average)
        # 左边距 0.1 => 0.8
        offset = int(average * 0.8)

        for im in images:
            w, h = im.size
            mask = im.convert('L').point(table)
            image.paste(im, (offset, int((self._height - h) / 2)), mask)
            offset = offset + w + random.randint(-rand, 0)

        if width > self._width:
            image = image.resize((self._width, self._height))

        return image

    def generate_image(self, chars):
        """Generate the image of the given characters.

        :param chars: text to be generated.
        """
        background = random_color(0, 255)
        image = self.create_background(background)
        color = tuple(map(lambda x: 255 - x, background))
        # for i in range(random.randint(8, 15)):
        for i in range(random.randint(3, 5)):
            self.draw_wave(
                image,
                [random.randint(0, int(IMAGE_WIDTH / 3)), random.randint(0, IMAGE_HEIGHT)],  # 横线起点都在左半边
                [random.randint(int(IMAGE_WIDTH / 3 * 2), IMAGE_WIDTH), random.randint(0, IMAGE_HEIGHT)],  # 横线终点
                random.randint(1, 10),  # sin函数频率， 数字越大波浪越多
                random.randint(1, 3),  # 振幅， 数字越大波浪越陡峭
                random.randint(1, 4),  # 宽度
                color if i % 3 == 0 else random_color(0, 255)
            )

        image = self.draw_text(image, chars, color)
        image = image.filter(ImageFilter.SMOOTH)
        return image


# 验证码一般都无视大小写；验证码长度4个字符
def random_captcha_text(char_set=number + alphabet + ALPHABET, captcha_size=MAX_CAPTCHA):
    ''' 指定使用的验证码内容列表和长期 返回随机的验证码文本 '''
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text


async def gen_captcha_text_and_image():
    '''生成字符对应的验证码 '''
    image = ImageCaptcha(width=IMAGE_WIDTH, height=IMAGE_HEIGHT,
                         fonts=['/Users/allovince/opt/htdocs/anti-captcha/fonts/Fangsong.ttf',
                                '/Users/allovince/opt/htdocs/anti-captcha/fonts/STHeiti.ttf'])  # 导入验证码包 生成一张空白图

    captcha_text = random_captcha_text()  # 随机一个验证码内容
    captcha_text = ''.join(captcha_text)  # 类型转换为字符串

    captcha = image.generate(captcha_text)
    # image.write(captcha_text, 'samples/' + captcha_text + '.jpg')  # 写到文件

    # rm  =  'rm '+captcha_text + '.jpg'
    # os.system(rm)

    captcha_image = Image.open(captcha)  # 转换为图片格式
    captcha_image = np.array(captcha_image)  # 转换为 np数组类型
    return captcha_text, captcha_image, captcha_image


async def main():
    text, image, original_image = await gen_captcha_text_and_image()
    # print(image[:, :, 0], 'a', image[:, :, 1])
    # print(image[:, :, 0] + image[:, :, 1])
    print(image[:, :, 0] + image[:, :, 1] + image[:, :, 2])
    print(np.mean(image, -1))
    # f = plt.figure()
    # plt.subplot(2, 2, 1), plt.imshow(original_image)
    # plt.xticks([]), plt.yticks([])
    # plt.subplot(2, 2, 2), plt.imshow(image)
    # plt.xticks([]), plt.yticks([])
    # plt.show()


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()
