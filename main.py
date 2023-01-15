import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.exceptions import NotFittedError

class TfIdfTransformer:
    """
    A class for performing TF-IDF transformation on text data.
    """
    def __init__(self):
        """
        Initializes an instance of the TfIdfTransformer class.
        """
        self.vectorizer = TfidfVectorizer()
        self.fitted = False

    def fit(self, data: list):
        """
        Fits the TfIdfTransformer to the input data.
        
        Parameters:
        - data: list of strings representing the text data to fit on.
        
        Returns:
        - None
        """
        self.vectorizer.fit(data)
        self.fitted = True

    def transform(self, data: list):
        """
        Transforms the input data using the fitted TfIdfTransformer.
        
        Parameters:
        - data: list of strings representing the text data to transform.
        
        Returns:
        - sparse matrix of the transformed data.
        """
        if not self.fitted:
            raise NotFittedError("TfIdfTransformer must be fitted before transforming data.")
        return self.vectorizer.transform(data)

