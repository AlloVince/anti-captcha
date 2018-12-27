from gen_captcha import gen_captcha_text_and_image, IMAGE_WIDTH, IMAGE_HEIGHT, MAX_CAPTCHA, CHAR_SET_LEN
import numpy as np
import tensorflow as tf
import os
import cv2
import glob
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# print("验证码文本最长字符数", MAX_CAPTCHA)  # 验证码最长4字符; 我全部固定为4,可以不固定. 如果验证码长度小于4，用'_'补齐


# 把彩色图像转为灰度图像（色彩对识别验证码没有什么用）
def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        # 上面的转法较快，正规转法如下
        # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img


"""
cnn在图像大小是2的倍数时性能最高, 如果你用的图像大小不是2的倍数，可以在图像边缘补无用像素。
np.pad(image【,((2,3),(2,2)), 'constant', constant_values=(255,))  # 在图像上补2行，下补3行，左补2行，右补2行
"""


def text2vec(text: str):
    text_len = len(text)
    if text_len > MAX_CAPTCHA:
        raise ValueError('超过验证码最长字符')

    vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)

    def char2pos(c):
        if c == '_':
            k = 62
            return k
        k = ord(c) - 48
        if k > 9:
            k = ord(c) - 55
            if k > 35:
                k = ord(c) - 61
                if k > 61:
                    raise ValueError('No Map')
        return k

    for i, c in enumerate(text):
        idx = i * CHAR_SET_LEN + char2pos(c)
        vector[idx] = 1
    return vector


# 向量转回文本
def vec2text(vec):
    char_pos = vec.nonzero()[0]
    text = []
    for i, c in enumerate(char_pos):
        char_at_pos = i  # c/63
        char_idx = c % CHAR_SET_LEN
        if char_idx < 10:
            char_code = char_idx + ord('0')
        elif char_idx < 36:
            char_code = char_idx - 10 + ord('A')
        elif char_idx < 62:
            char_code = char_idx - 36 + ord('a')
        elif char_idx == 62:
            char_code = ord('_')
        else:
            raise ValueError('error')
        text.append(chr(char_code))
    return "".join(text)


"""
#向量（大小MAX_CAPTCHA*CHAR_SET_LEN）用0,1编码 每63个编码一个字符，这样顺利有，字符也有
vec = text2vec("F5Sd")
text = vec2text(vec)
print(text)  # F5Sd
vec = text2vec("SFd5")
text = vec2text(vec)
print(text)  # SFd5
"""


# 生成一个训练batch
def get_next_batch(batch_size=128):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])

    # 有时生成图像大小不是(60, 160, 3)
    # def wrap_gen_captcha_text_and_image():
    #     ''' 获取一张图，判断其是否符合（60，160，3）的规格'''
    #     while True:
    #         text, image, o = gen_captcha_text_and_image()
    #         if image.shape == (IMAGE_HEIGHT, IMAGE_WIDTH, 3):  # 此部分应该与开头部分图片宽高吻合
    #             return text, image

    for i in range(batch_size):
        text, image, o = gen_captcha_text_and_image()
        # image = convert2gray(image)

        batch_x[i, :] = image.flatten() / 255  # (image.flatten()-128)/128  mean为0
        batch_y[i, :] = text2vec(text)
    # 返回该训练批次
    return batch_x, batch_y


####################################################################
# 申请占位符 按照图片
X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN])
keep_prob = tf.placeholder(tf.float32)  # dropout


# 定义CNN
def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
    # 将占位符 转换为 按照图片给的新样式
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    # w_c1_alpha = np.sqrt(2.0/(IMAGE_HEIGHT*IMAGE_WIDTH)) #
    # w_c2_alpha = np.sqrt(2.0/(3*3*32))
    # w_c3_alpha = np.sqrt(2.0/(3*3*64))
    # w_d1_alpha = np.sqrt(2.0/(8*32*64))
    # out_alpha = np.sqrt(2.0/1024)

    # 3 conv layer
    w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32]))  # 从正太分布输出随机值
    b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob)

    w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)

    w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)

    # Fully connected layer
    w_d = tf.Variable(w_alpha * tf.random_normal([10 * 25 * 64, 1024]))  # 10 * 25 * 64是由conv3得到的
    b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
    dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)

    w_out = tf.Variable(w_alpha * tf.random_normal([1024, MAX_CAPTCHA * CHAR_SET_LEN]))
    b_out = tf.Variable(b_alpha * tf.random_normal([MAX_CAPTCHA * CHAR_SET_LEN]))
    out = tf.add(tf.matmul(dense, w_out), b_out)
    # out = tf.nn.softmax(out)
    return out


model_path = os.path.dirname(os.path.realpath(__file__)) + '/models'
model_name = model_path + '/model'
ACC_TARGET = float(os.environ.get('ACC_TARGET', 0.95))
image_path = os.path.dirname(os.path.realpath(__file__)) + '/samples'
logs_path = os.path.dirname(os.path.realpath(__file__)) + '/logs'


def parse_image(img_path):
    def _parse_image(_image_path: bytes):
        original_image = cv2.imread(_image_path.decode('utf-8'))
        image = cv2.medianBlur(original_image, 5)
        ret, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = image.shape
        revert = image.copy()
        count = 0
        for i in range(height):
            for j in range(width):
                if image[i, j] == 0:
                    count += 1
                revert[i, j] = (255 - image[i, j])
        image = revert if count > height * width / 2 else image
        text = os.path.splitext(os.path.basename(_image_path))[0]
        text = text.decode('utf-8')
        # print(image.flatten() / 255, width, height, \
        #        text2vec(text), text, _image_path)
        return image.flatten() / 255, width, height, \
               text2vec(text), text, _image_path.decode('utf-8')

    return tuple(tf.py_func(_parse_image, [img_path], [tf.double, tf.int64, tf.int64, tf.double, tf.string, tf.string]))


def get_element():
    images = lambda: glob.iglob(image_path + '/*.jpg')
    while True:
        try:
            images().__next__()
            print('Detected images')
            break
        except StopIteration:
            print('Not found image, sleeping...')
            time.sleep(5)

    batch_size = 1
    dataset = tf.data.Dataset.from_generator(images, output_types=(tf.string))
    dataset = dataset.map(parse_image, num_parallel_calls=batch_size)
    dataset.prefetch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


# 训练
def train_crack_captcha_cnn():
    output = crack_captcha_cnn()
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, Y))
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
    # 最后一层用来分类的softmax和sigmoid有什么不同？
    # optimizer 为了加快训练 learning_rate应该开始大，然后慢慢衰
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()
    checkpoint = tf.train.latest_checkpoint(model_path)

    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy", accuracy)
    merged_summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if checkpoint:
            saver.restore(sess, checkpoint)
            useless, step = checkpoint.split('model-')
            step = int(step)
            print('找到 Checkpoint %s, 继续上次训练 Step %s' % (checkpoint, step))
        else:
            step = 1
            print('开始新的训练 Step', step)

        op = get_element()
        while True:
            # batch_x, batch_y = get_next_batch(64)
            x, y, w, h, t, p = sess.run(op)
            _, loss_, summary = sess.run([optimizer, loss, merged_summary_op],
                                         feed_dict={X: x, Y: y, keep_prob: 0.75})
            summary_writer.add_summary(summary, step)
            print('Loss', step, loss_)

            # 每100 step计算一次准确率
            # if step % 100 == 0:
            #     batch_x_test, batch_y_test = get_next_batch(100)
            #     acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
            #     print('ACC', step, acc)
            #     saver.save(sess, model_name, global_step=step)
            #     print('model saved to', model_name)
            #     if acc > ACC_TARGET:
            #         break
            step += 1


if __name__ == '__main__':
    train_crack_captcha_cnn()
