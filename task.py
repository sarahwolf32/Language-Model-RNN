import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
from model import Model
import argparse

'''
Use eager execution to train my vanilla-RNN.

My Code Wish List:
    - write loss to tensorboard (use eager summary, might need internet)
    - save model (inherit from keras???) - need internet to research
    - fix one-hot shapes/types (a little messier than needbe)
'''

tf.enable_eager_execution()

def sample(n, variables, model, code_map, C):
    print("drawing samples!")
    for i in range(n):
        word = model.sample_word(variables, code_map, C)
        print(word)

def train(config):

    model = Model(config)

    # import words
    words = model.words_list()
    print("Found " + str(len(words)) + " words!")

    # create dataset
    dataset = model.create_dataset(words)
    character_map, code_map = model.character_maps(words)
    C = len(character_map)

    # create variables
    variables, variables_list = model.create_weights(C)

    # create optimizer
    optimizer = tf.train.GradientDescentOptimizer(config.learning_rate)

    # train loop
    print("\nSamples after no training:")
    sample(3, variables, model, code_map, C)
    for epoch in range(config.num_epochs):

        losses = []
        for tensor in dataset.make_one_shot_iterator():
            word = tensor.numpy()
            y = model.vectorize_word(word, character_map, C)
            with tf.GradientTape() as tape:
                y_pred = model.input_word(y, variables, C)

                # get loss
                y_with_end = model.add_end_tag(y, C, character_map)
                loss = model.compute_loss(tf.cast(y_with_end, tf.float32), y_pred)
                losses.append(loss.numpy())
                #print(word + ": loss = " + str(loss.numpy()))

                # get gradients
                grads = tape.gradient(loss, variables_list)
                optimizer.apply_gradients(zip(grads, variables_list))

        # print epoch stats
        print("\nepoch " + str(epoch))
        sample(3, variables, model, code_map, C)
        avg_loss = np.mean(losses)
        print("avg_loss = " + str(avg_loss))





if __name__=='__main__':

    # config arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='elvish_words.txt')
    parser.add_argument('--num-epochs', type=int, default=10)
    parser.add_argument('--nodes', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--weights-init-stddev', type=float, default=0.2)
    config = parser.parse_args()

    train(config)







