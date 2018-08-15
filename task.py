import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
from model import Model
import argparse
import pickle
import os

'''
Use eager execution to train my vanilla-RNN.

My Code Wish List:
    - write loss to tensorboard (use eager summary, might need internet)
    - fix one-hot shapes/types (a little messier than needbe)
    - fix non-standard character display
'''



tf.enable_eager_execution()

def sample(n, variables, model, code_map, C):
    for i in range(n):
        word = model.sample_word(variables, code_map, C)
        print(word)

def save_model_structure(config, code_map):
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)
    save_dict = {'nodes': config.nodes, 'code_map': code_map}
    save_path = config.save_dir + '/model-structure.pickle'
    with open(save_path, 'wb') as handle:
        pickle.dump(save_dict, handle)

def train(config):

    model = Model(config)

    # import words
    words = model.words_list()
    print("Found " + str(len(words)) + " words!")
    dataset = model.create_dataset(words)
    character_map, code_map = model.character_maps(words)
    save_model_structure(config, code_map)
    C = len(character_map)

    # create variables
    variables, variables_list = model.create_weights(C)
    saver = tfe.Saver(variables)
    optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate)

    # track best
    lowest_loss = None
    lowest_loss_epoch = None

    # train loop
    print("\nSamples after no training:")
    sample(3, variables, model, code_map, C)
    for epoch in range(config.num_epochs):

        losses = []
        for tensor in dataset.make_one_shot_iterator():
            word = tensor.numpy().decode('utf-8')
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
        sample(2, variables, model, code_map, C)
        avg_loss = np.mean(losses)
        if (lowest_loss == None) or (avg_loss < lowest_loss):
            lowest_loss = avg_loss
            lowest_loss_epoch = epoch
            save_path = config.save_dir + '/checkpoint.ckpt'
            saver.save(save_path, global_step=epoch)
        print("avg_loss = " + str(avg_loss))

    # print final stats
    print("\nLowest loss = " + str(lowest_loss) + ", epoch " + str(lowest_loss_epoch))

def load_model(config):

    # unpickle structure
    filename = config.model_dir + '/model-structure.pickle'
    with open(filename, 'rb') as handle:
        structure = pickle.load(handle)
    config.nodes = structure['nodes']
    code_map = structure['code_map']
    C = len(code_map)

    # load variables
    model = Model(config)
    variables, variables_list = model.create_weights(C)
    saver = tfe.Saver(variables_list)
    model_path = str(tf.train.latest_checkpoint(config.model_dir))
    saver.restore(model_path)

    # draw samples
    sample(config.num_samples, variables, model, code_map, C)


if __name__=='__main__':

    # config arguments
    parser = argparse.ArgumentParser()

    # train arguments
    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--data-dir', default='word_lists/elvish_words.txt')
    parser.add_argument('--num-epochs', type=int, default=200)
    parser.add_argument('--nodes', type=int, default=80)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--weights-init-stddev', type=float, default=0.2)
    parser.add_argument('--save-dir', default='checkpoints')

    # sample arguments
    parser.add_argument('--model-dir', default='models/dinosaur_names_model')
    parser.add_argument('--output-dir', default='output')
    parser.add_argument('--num-samples', type=int, default=100)
    config = parser.parse_args()

    if config.train:
        train(config)
    else:
        load_model(config)








