import unittest
import numpy as np
from main import TfIdfTransformer
from sklearn.exceptions import NotFittedError

class TestTfIdfTransformer(unittest.TestCase):
    def setUp(self):
        self.data = ["this is a test", "this is another test", "test test test"]
        self.transformer = TfIdfTransformer()
        self.transformer.fit(self.data)

    def test_transform(self):
        transformed_data = self.transformer.transform(self.data)
        # Assert that the shape of the transformed data is (3, 7)
        self.assertEqual(transformed_data.shape, (3, 7))
        
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

if __name__ == '__main__':
    unittest.main()
