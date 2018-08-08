import unittest
import model
import tensorflow as tf
import numpy as np

class Tests(unittest.TestCase):

    def test_input_letter(self):

        # setup
        n = 10
        c = 5
        prev_y = np.zeros((1, c))
        prev_y[0][0] = 1.
        prev_a = np.random.rand(1, n)
        vars_dict = {
            'Wa': np.random.rand(n + c, n),
            'ba': np.random.rand(1, n),
            'Wy': np.random.rand(n, c),
            'by': np.random.rand(1, c)}

        # run
        y_pred_var, a_var = model.input_letter(prev_y, prev_a, vars_dict)
        sess = tf.Session()
        y_pred, a = sess.run([y_pred_var, a_var])

        # verify
        self.assertEqual(y_pred.shape, (1, c))
        self.assertEqual(a.shape, (1, n))

    def test_character_maps(self):
        words = ["Galadriel", "Elrond", "Evenstar", "Bilbo Baggins"]
        character_map, code_map = model.character_maps(words)

        # test character map {char: Int}
        self.assertTrue('G' in character_map)
        self.assertTrue(' ' in character_map)
        self.assertFalse('w' in character_map)

        # test code map {Int: char}
        self.assertEqual(code_map[character_map['G']], 'G')
        self.assertEqual(len(character_map), len(code_map))

    def test_vectorize_word(self):
        words = ["Galadriel", "Elrond"]
        character_map, _ = model.character_maps(words)
        C = len(character_map)
        word = "Elrond"
        word_vecs = model.vectorize_word(word, character_map, C)
        self.assertEqual(word_vecs.shape, (len(word), C))

        



if __name__ == '__main__':
    unittest.main()