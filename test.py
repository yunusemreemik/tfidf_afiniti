import unittest
import numpy as np
from TfidfTransformer_afiniti import TfIdfTransformer
from sklearn.exceptions import NotFittedError

class TestTfIdfTransformer(unittest.TestCase):
    def setUp(self):
        self.data = ["this is a test", "this is another test", "test test test"]
        #self.data = ["this is some text", "text is great", "this is more text"]
        self.transformer = TfIdfTransformer()
        self.transformer.fit(self.data)

    def test_not_fitted_error(self):
        # Create a new instance of TfIdfTransformer without fitting
        not_fitted_transformer = TfIdfTransformer()
        # Assert that a NotFittedError is raised when trying to transform data
        with self.assertRaises(NotFittedError):
            not_fitted_transformer.transform(self.data)

    def test_transform_with_new_data(self):
        new_data = ["this is a new test"]
        transformed_data = self.transformer.transform(new_data)
        # Assert that the shape of the transformed data is (1, 7)
        self.assertEqual(transformed_data.shape, (1, 7))

    def test_transform(self):

        self.transformer.fit_transform(self.data)

        new_data = ["this is even more text"]
        tfidf_new_data = self.transformer.transform(new_data)
        
        # check if the shape of the data is correct
        self.assertEqual(tfidf_new_data.shape[0], 1)
        self.assertEqual(tfidf_new_data.shape[1], 4)

        # check if the new data is correctly appended to the existing data
        self.assertEqual(len(tfidf_new_data.data), 4)
        self.assertEqual(tfidf_new_data.data[-1], "this is even more text")


if __name__ == '__main__':
    unittest.main()
