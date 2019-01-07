# -*-coding:utf-8-*-

import tensorflow as tf
import eva_captcha
import eva_model


def main():
    captcha_gen = eva_captcha.captcha_tensorflow()
    width, height, char_num, characters, classes = captcha_gen.get_parameter()
    # 定义输入
    img_x = tf.placeholder(tf.float32, [None, height, width, 1])
    img_y = tf.placeholder(tf.float32, [None, char_num * classes])
    # 获取模型
    captcha_mod = eva_model.create_model()
    pre_y = captcha_mod.inference(img_x)
    # 定义损失函数及优化方法
    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=img_y, logits=pre_y))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    # 计算准确率
    predict = tf.reshape(pre_y, [-1, char_num, classes])
    real = tf.reshape(img_y, [-1, char_num, classes])
    correct_prediction = tf.equal(tf.argmax(predict, 2), tf.argmax(real, 2))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
    # 收集变量的值
    tf.summary.scalar("cross_entropy", cross_entropy)
    tf.summary.scalar("accuracy", accuracy)
    tf.summary.image("input", img_x)
    merged_summary = tf.summary.merge_all()
    # 启动Session
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter("../../log/captcha/01", graph=sess.graph)
        step = 0
        while True:
            batch_x, batch_y = captcha_gen.get_batch_captcha(64)
            _, loss, summary = sess.run([train_step, cross_entropy, merged_summary],
                                        feed_dict={img_x: batch_x, img_y: batch_y})
            writer.add_summary(summary, step)
            # print ('step:%d,loss:%f' % (step,loss))
            if step % 100 == 0:
                batch_x_test, batch_y_test = captcha_gen.get_batch_captcha(100)
                loss, acc = sess.run([cross_entropy, accuracy], feed_dict={img_x: batch_x_test, img_y: batch_y_test})
                print('step:%d, loss:%f, accuracy:%f' % (step, loss, acc))
                if acc > 0.1:
                    saver.save(sess, "model/01/eva_model.ckpt")
                    break
            step += 1


if __name__ == '__main__':
    main()
