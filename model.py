import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from random import shuffle

'''
What can this model do?
    1. Generate new Elvish names.
    2. Score words on how 'Elvish' they are.
'''

class Model:

    def __init__(self, config):
        self.config = config

    # load words

    def words_list(self):
        '''
        filename: Path to a .txt file containing a list of words. One word per line, no commas.
        returns: A shuffled list of the words in that file.
        '''
        with open(self.config.data_dir) as f:
            words = f.readlines()
        words = [w.rstrip().lower() for w in words]
        shuffle(words)
        return words

    def create_dataset(self, words_list):
        '''
        words_list: A python list of strings.
        returns: A Tensorflow dataset of those strings.
        '''
        words = tf.constant(words_list)
        dataset = tf.data.Dataset.from_tensor_slices(words)
        return dataset

    # build character maps

    def character_maps(self, words_list):
        i = 0
        character_map = {}

        # assign code to each unique character
        for word in words_list:
            for char in word:
                if not char in character_map:
                    character_map[char] = i
                    i += 1

        # add any special codes
        character_map['<END>'] = i

        # reverse dictionary to look up in other direction
        code_map = {v: k for k, v in character_map.iteritems()}

        return character_map, code_map

    def one_hot(self, code, C):
        '''A single one-hot vector for one letter-code.'''
        x = np.zeros(C)
        x[code] = 1.
        return x

    def vectorize_word(self, word, character_map, C):
        codes = [character_map[letter] for letter in word]
        vectors = np.array([self.one_hot(code, C) for code in codes])
        return vectors

    def add_end_tag(self, y, C, character_map):
        '''
        Y by default does not include the '<END>' tag. 
        It needs to be added to y, for computing the loss.
        '''
        end_code_vec = self.one_hot(character_map['<END>'], C)
        end_code_vec = tf.reshape(end_code_vec, shape=[1,-1])
        y_extended = tf.concat([y, end_code_vec], axis=0)
        return y_extended

    # build graph

    def create_weights(self, C):
        n = self.config.nodes
        std = self.config.weights_init_stddev
        Wa = tfe.Variable(tf.random_normal([n + C, n], stddev=std), name='Wa')
        ba = tfe.Variable(tf.zeros([1, n]), name='ba')
        Wy = tfe.Variable(tf.random_normal([n, C], stddev=std), name='Wy')
        by = tfe.Variable(tf.zeros([1, C]), name='by')
        variables = {'Wa': Wa, 'ba': ba, 'Wy': Wy, 'by': by}
        variables_list = variables.values()
        return variables, variables_list

    def input_letter(self, prev_y, prev_a, var):
        '''
        prev_y: A placeholder for y<t-1>, the previous character. A one-hot vector of shape [1, c]
        prev_a: A tensor for a<t-1>, the previous activation. Shaped [1, n]
        var: Dictionary of trainable weights and biases
            Wa: [n+c, n]
            ba: [1, n]
            Wy: [n, c]
            by: [1, c]
        returns: y<t> and a<t>
        '''
        Wa, ba, Wy, by = var['Wa'], var['ba'], var['Wy'], var['by']
        inputs = tf.concat([prev_y, prev_a], axis=1)
        a = tf.nn.tanh(tf.matmul(inputs, Wa) + ba)
        #y_pred = tf.nn.sigmoid(tf.matmul(a, Wy) + by)
        y_pred = tf.nn.softmax(tf.matmul(a, Wy) + by)
        return (y_pred, a)

    def input_word(self, y, var, C):
        '''
        y: All the one-hot vectors for the letters in the word, shaped [word_len, C]
        var: Dictionary of trainable weights and biases
        returns: y_pred, shaped [word_len + 1, C]  
        '''

        # x = [0, y<1>, y<2>,...,y<t>]
        zeros = tf.zeros([1, C])
        x = tf.concat([zeros, y], axis=0)

        # initialize a<0>
        a_t = np.zeros((1, self.config.nodes))

        # sequentially input letters
        y_pred = []
        for t in range(x.shape[0]):
            x_t = tf.reshape(x[t],[1, -1])
            y_pred_t, a_t =  self.input_letter(x_t, a_t, var)
            y_pred.append(y_pred_t[0]) 

        y_pred = tf.stack(y_pred)
        return y_pred

    def compute_loss(self, y, y_pred):
        '''
        y: One hot vectors for correct letter values. Shaped [word_len + 1, C].
        y_pred: Predicted probability distribution for letter values. Shaped [word_len + 1, C].
        '''
        loss = -tf.reduce_sum(y * tf.log(y_pred))
        return loss

    # sample

    def sample_word(self, variables, code_map, C, max_length=20):
        word = ''
        x = tf.zeros([1, C])
        a = np.zeros((1, self.config.nodes))

        while True:
            x, a = self.input_letter(x, a, variables)
            y_probs = x.numpy()[0]

            # make a random letter choice weighted by y_probs
            code = np.random.choice(len(y_probs), p=y_probs)
            letter = code_map[code]

            # break if word is finished or exceeded length limit
            if (letter == '<END>') or (len(word) >= max_length):
                break

            # add letter to word
            word += letter

        return word






