import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
from model import Model
import argparse
import pickle
import os

'''
TODO:
- write loss to tensorboard
'''

tf.enable_eager_execution()

def sample(n, variables, model, code_map, C, filename, config):

    # make sure output directory exists
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    # write to save path
    save_path = config.output_dir + "/" + filename
    f = open(save_path, 'w')
    for i in range(n):
        word = model.sample_word(variables, code_map, C)
        f.write(word + "\n")
    f.close()

def save_model_structure(config, code_map):
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)
    save_dict = {'nodes': config.nodes, 'code_map': code_map}
    save_path = config.save_dir + '/model-structure.pickle'
    with open(save_path, 'wb') as handle:
        pickle.dump(save_dict, handle)

def log_loss(loss, global_step):
    global_step.assign_add(1)
    with tf.contrib.summary.always_record_summaries():
        tf.contrib.summary.scalar('per-epoch mean loss', loss)

def train(config):

    model = Model(config)

    # import words
    words = model.words_list()
    dataset = model.create_dataset(words)
    character_map, code_map = model.character_maps(words)
    save_model_structure(config, code_map)
    C = len(character_map)

    # create variables
    variables, variables_list = model.create_weights(C)

    # setup saving and logging
    saver = tfe.Saver(variables)
    summary_writer = tf.contrib.summary.create_file_writer(config.log_dir, flush_millis=10000)
    summary_writer.set_as_default()
    global_step = tf.train.get_or_create_global_step()

    optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate)

    # track best
    lowest_loss = None
    lowest_loss_epoch = None

    # train loop
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

                # get gradients
                grads = tape.gradient(loss, variables_list)
                optimizer.apply_gradients(zip(grads, variables_list))

        # log and save if best avg_loss so far
        avg_loss = np.mean(losses)
        log_loss(avg_loss, global_step)

        if (lowest_loss == None) or (avg_loss < lowest_loss):
            lowest_loss = avg_loss
            lowest_loss_epoch = epoch
            save_path = config.save_dir + '/checkpoint.ckpt'
            saver.save(save_path, global_step=epoch)

        # print epoch loss
        print("epoch " + str(epoch) + ": avg_loss = " + str(avg_loss))

    # print final best loss
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
    sample(config.num_samples, variables, model, code_map, C, 'sample.txt', config)


if __name__=='__main__':

    # config arguments
    parser = argparse.ArgumentParser()

    # train arguments
    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--data-dir', default='word_lists/elvish_words.txt')
    parser.add_argument('--num-epochs', type=int, default=150)
    parser.add_argument('--nodes', type=int, default=80)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--weights-init-stddev', type=float, default=0.2)
    parser.add_argument('--save-dir', default='checkpoints')
    parser.add_argument('--log-dir', default='logs')

    # sample arguments
    parser.add_argument('--model-dir', default='models/dinosaur_names_model')
    parser.add_argument('--output-dir', default='output')
    parser.add_argument('--num-samples', type=int, default=100)
    config = parser.parse_args()

    if config.train:
        train(config)
    else:
        load_model(config)








