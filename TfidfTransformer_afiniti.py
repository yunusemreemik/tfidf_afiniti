from sklearn.feature_extraction.text import TfidfVectorizer

class TfIdfTransformer:
    def __init__(self):
        self.tfidf = TfidfVectorizer()

    def fit(self, data):
        return self.tfidf.fit(data)

    def transform(self, data):
        # check if the vectorizer has been fitted
        try:
            getattr(self.tfidf, 'vocabulary_')
        except AttributeError:
            print("The vectorizer must be fitted before adding new data.")
            return
        return self.tfidf.transform(data)
    
    def fit_transform(self, data):
        self.tfidf.fit(data)
        return self.tfidf.transform(data)
    
    def add_new_data(self, new_data):
        # check if new_data is not None or empty
        if not new_data:
            print("No new data provided.")
            return
        
        # check if new_data is a list
        if not isinstance(new_data, list):
            print("New data must be provided in the form of a list.")
            return
        
        # check if new_data is a list of strings
        if not all(isinstance(i, str) for i in new_data):
            print("New data must be a list of strings.")
            return
        
        # check if the vectorizer has been fitted
        try:
            getattr(self.tfidf, 'vocabulary_')
        except AttributeError:
            print("The vectorizer must be fitted before adding new data.")
            return
        
        # append new_data to the existing data
        data = self.tfidf.transform(new_data)
        return data

if __name__=='__main__':
    # a valid case
    data = ["This is the first document.", "This is the second document."]
    tfidf_transformer = TfIdfTransformer()
    tfidf_matrix = tfidf_transformer.fit_transform(data)
    
    new_data = ["This is the third document."]
    added_tfidf_matrix = tfidf_transformer.add_new_data(new_data)

    # test case 1 (check none or empty)
    new_tfidf_transformer = TfIdfTransformer()
    new_tfidf_matrix = new_tfidf_transformer.fit_transform(data)
    new_data = []
    added_tfidf_matrix = new_tfidf_transformer.add_new_data(new_data)

    # test case 2 (check if new_data is a list of strings)
    new_tfidf_transformer = TfIdfTransformer()
    new_tfidf_matrix = new_tfidf_transformer.fit_transform(data)
    new_data = "This is the third document."
    added_tfidf_matrix = new_tfidf_transformer.add_new_data(new_data)

    # test case 3 (check the vectorizer has been fitted)
    new_tfidf_transformer = TfIdfTransformer()
    new_data = "This is the third document."
    added_tfidf_matrix = new_tfidf_transformer.add_new_data(new_data)


    pass