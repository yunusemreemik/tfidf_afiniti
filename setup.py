# setup.py

from setuptools import setup
long_description = ""
"""
with open("README.md", "r") as fh:
    long_description = fh.read()
 """
setup(
    name='tfidf_transformer_afiniti',
    version='0.1',
    description='A package for performing TF-IDF transformation on text data',
    author='Yunus Emre Emik',
    author_email='yunus_emre3497@hotmail.com',
    url='https://github.com/yunusemreemik/tfidf_afiniti.git',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=['tfidf_transformer_afiniti'],
    install_requires=[
        'scikit-learn',
    ],
)

