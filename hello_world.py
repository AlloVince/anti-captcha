import tensorflow as tf


def main():
    hello = tf.constant('hello TF')
    sess = tf.Session()
    print(sess.run(hello))


if __name__ == '__main__':
    main()
