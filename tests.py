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


if __name__ == '__main__':
    unittest.main()