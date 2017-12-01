from __future__ import print_function
import numpy as np
import tensorflow as tf
import heapq
import argparse
import os
import heapq

from six.moves import cPickle
from collections import namedtuple

from model import Model

def topk(arr, count):
    return heapq.nlargest(count, range(len(arr)), arr.take)

Term = namedtuple('Term', ['name', 'probability'])

class TermDistribution:
    def __init__(self, words, vocab, probs):
        self.words = words
        self.vocab = vocab
        self.probs = probs
    
    def get(self, key, default):
        if key in self.vocab:
            return self[key]
        else:
            return default
    
    def __contains__(self, key):
        return key in self.vocab
    
    def __getitem__(self, key):
        return self.probs[self.vocab[key]]
    
    def keys(self):
        return self.words
    
    def __iter__(self):
        for (k,v) in self.vocab:
            yield (k, self.probs[v])

    def topk(self, count):
        for x in topk(self.probs, count):
            yield Term(self.words[x], self.probs[x])

    def __repr__(self):
        return dict(self)

Step = namedtuple('Step', ['name', 'probabilities'])

def step(model, sess, words, vocab, elems):
    """
    Step through the input sequence and yield the probabilities associated with
    each term.
    """
    state = sess.run(model.cell.zero_state(1, tf.float32))
    for (word, next_word) in zip(elems,elems[1:]):
        x = np.zeros((1, 1))
        x[0, 0] = vocab.get(word,0)
        feed = {model.input_data: x, model.initial_state:state}
        [probs, state] = sess.run([model.probs, model.final_state], feed)
        yield Step(next_word, TermDistribution(words, vocab, probs[0]))

def do_spellcheck(model, sess, words, vocab, elems, treshold=0.15, suggestion_count=3):
    for (idx, (word, probs)) in enumerate(step(model, sess, words, vocab, elems), 1):
        prob = probs.get(word, 0.0)
        probs = (x for x in probs.topk(suggestion_count) if x.name != word and x.probability - prob > treshold)
        sugg = ", ".join("%r (+%.1f%%)" % (x.name, (x.probability - prob) * 100) for x in probs)
        if len(sugg) > 0:
            print("Call #%d: %s" % (idx, word), "%.1f%%" % (prob * 100), "did you mean", sugg)

################################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='save',
                       help='model directory to load stored checkpointed models from')
    parser.add_argument('-q', nargs='+', dest="query",
                       help='Spell-check a sequence of function calls.')

    parser.add_argument("-l", "--list", dest="vocabs", type=str, default=None, help="Show vocabs starting with")

    args = parser.parse_args()
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(args.save_dir, 'words_vocab.pkl'), 'rb') as f:
        words, vocab = cPickle.load(f)

    # 1. Query vocabs
    if args.vocabs is not None:
        result = list(x for x in words if x.startswith(args.vocabs))
        result.sort()
        print(result)
        return

    # 2. Spell-check
    model = Model(saved_args, True)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            do_spellcheck(model, sess, words, vocab, args.query)

if __name__ == '__main__':
    main()

