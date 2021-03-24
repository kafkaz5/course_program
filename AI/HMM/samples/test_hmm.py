import unittest
import HMM1

class TestStringMethods(unittest.TestCase):

    def test_hmm(self):
        A = list(map(float,'4 4 0.0 0.8 0.1 0.1 0.1 0.0 0.8 0.1 0.1 0.1 0.0 0.8 0.8 0.1 0.1 0.0'.split(' ')))
        B = list(map(float,'4 4 0.9 0.1 0.0 0.0 0.0 0.9 0.1 0.0 0.0 0.0 0.9 0.1 0.1 0.0 0.0 0.9'.split(' ')))
        initial = list(map(float,'1 4 1.0 0.0 0.0 0.0'.split(' ')))
        obs_input = list(map(int,'8 0 1 2 3 0 1 2 3'.split(' ')))
        actual = HMM1.get_alpha(A=A, B=B, initial=initial, obs_input=obs_input)
        self.assertEqual(actual, 0.090276)

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

if __name__ == '__main__':
    unittest.main()