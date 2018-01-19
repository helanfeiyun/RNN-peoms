from __future__ import print_function
import numpy as np
import tensorflow as tf

import time
import os,sys
from six.moves import cPickle

from model.model_rnn import Model
from utils.parser import parse_train_args
from utils.loader import TextLoader

def main():
    args = parse_train_args()
    train(args)

def train(args):
    data_loader = TextLoader(args.batch_size)
    args.vocab_size = data_loader.vocab_size

    # check compatibility if training is continued from previously saved model
    # if args.init_from is not None:
    #     # check if all necessary files exist
    #     assert os.path.isdir(args.init_from)," %s must be a a path" % args.init_from
    #     assert os.path.isfile(os.path.join(args.init_from,"config.pkl")),"config.pkl file does not exist in path %s"%args.init_from
    #     assert os.path.isfile(os.path.join(args.init_from,"chars_vocab.pkl")),"chars_vocab.pkl.pkl file does not exist in path %s" % args.init_from
    #     ckpt = tf.train.get_checkpoint_state(args.init_from)
    #     assert ckpt,"No checkpoint found"
    #     assert ckpt.model_checkpoint_path,"No model path found in checkpoint"
    #     assert os.path.isfile(os.path.join(args.init_from,"iterations")),"iterations file does not exist in path %s " % args.init_from
    #
    #     # open old config and check if models are compatible
    #     with open(os.path.join(args.init_from, 'config.pkl'),'rb') as f:
    #         saved_model_args = cPickle.load(f)
    #     need_be_same=["model","rnn_size","num_layers"]
    #     for checkme in need_be_same:
    #         assert vars(saved_model_args)[checkme]==vars(args)[checkme],"Command line argument and saved model disagree on '%s' "%checkme
    #
    #     # open saved vocab/dict and check if vocabs/dicts are compatible
    #     with open(os.path.join(args.init_from, 'chars_vocab.pkl'),'rb') as f:
    #         saved_chars, saved_vocab = cPickle.load(f)
    #     assert saved_chars==data_loader.chars, "Data and loaded model disagree on character set!"
    #     assert saved_vocab==data_loader.vocab, "Data and loaded model disagree on dictionary mappings!"


    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        cPickle.dump(args, f)
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'wb') as f:
        cPickle.dump((data_loader.chars, data_loader.vocab), f)

    model = Model(args)

    with tf.Session() as sess:
        # instrument for tensorboard
        summaries = tf.summary.merge_all()
        writer = tf.summary.FileWriter(
            os.path.join('F:/1.0python/9-22/log', time.strftime("%Y-%m-%d-%H-%M-%S")))
        writer.add_graph(sess.graph)

        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        iterations = 0
        # restore model and number of iterations
        #if args.init_from is not None:
        saver.restore(sess, 'F:/1.0python/9-22/save/model.ckpt-6001')
        with open(os.path.join(args.save_dir, 'iterations'),'rb') as f:
            iterations = cPickle.load(f)
        losses = []
        for e in range(args.num_epochs):
            sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
            data_loader.reset_batch_pointer()
            for b in range(data_loader.num_batches):
                iterations += 1
                start = time.time()
                x, y = data_loader.next_batch()
                feed = {model.input_data: x, model.targets: y}
                train_loss, _ , _ = sess.run([model.cost, model.final_state, model.train_op], feed)
                end = time.time()
                sys.stdout.write('\r')
                info = "{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                    .format(e * data_loader.num_batches + b,
                            args.num_epochs * data_loader.num_batches,
                            e, train_loss, end - start)
                sys.stdout.write(info)
                sys.stdout.flush()
                losses.append(train_loss)
                if (e * data_loader.num_batches + b) % args.save_every == 0\
                    or (e==args.num_epochs-1 and b == data_loader.num_batches-1): # save for the last result
                    checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step = iterations)
                    with open(os.path.join(args.save_dir,"iterations"),'wb') as f:
                        cPickle.dump(iterations,f)
                    with open(os.path.join(args.save_dir,"losses-"+str(iterations)),'wb') as f:
                        cPickle.dump(losses,f)
                    losses = []
                    sys.stdout.write('\n')
                    print("model saved to {}".format(checkpoint_path))
            sys.stdout.write('\n')

if __name__ == '__main__':
    main()
