# TFIDF Transformer Afiniti

A package for performing TF-IDF transformation on text data.
This project developing for an assessment:

Create a framework that does tf-idf transformation, you can use sklearn's tfidf function.
Keep the following in mind

* Handle Edge Cases : what happens when new text data arrives
* Create Unit Tests : check failure scenarios
* Add Docstrings : assume you will hand this code to some other SWE
* Obey Engineering Best Practices
* Use necessary inheritances

Create a (pypi) package out of this framework.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install.

```bash
pip install tfidf-transformer-afiniti
```

## Usage

```python
from tfidf_transformer_afiniti.main import TfidfFramework

framework = TfidfFramework()

# Append some data to the data list
data = ["This is the first document.", "This document is the second document.", "And this is the third one."]
for d in data:
    framework.append_data(d)

# Print the tf-idf matrix
print(framework.tfidf_matrix.toarray())

# Add new one
new_data = "this is a new test document"
framework.append_data(new_data)
# Print the tf-idf matrix
print(framework.tfidf_matrix.toarray())

# Add new list
new_list_data = ["And this is the realy fifth one.","And this is the finaly sixth one."]  
framework.append_list_data(new_list_data)
# Print the tf-idf matrix
print(framework.tfidf_matrix.toarray())
```

## Usage

```bash
python -m unittest tfidf_transformer_afiniti/tfidf_test.py
```