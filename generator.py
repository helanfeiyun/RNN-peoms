#-*- coding:utf-8 -*-

from __future__ import print_function
import tensorflow as tf
import os
from six.moves import cPickle

from model.model_rnn import Model
from utils.parser import parse_generator_args

def main():
    args = parse_generator_args()
    generate(args)

def generate(args):
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'rb') as f:
        chars, vocab = cPickle.load(f)
    model = Model(saved_args, True)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print(model.sample(sess, chars, vocab, args.prime, args.sample))

if __name__ == '__main__':
    main()
