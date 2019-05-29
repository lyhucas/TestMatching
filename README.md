# TestMatching
We organized this reposity mainly for testing our fingerprint matching algorithm

# Requirements 
tensorflow 1.x
Python 3.x

# Usage
Extract fingerprint features
## python match/extract_feature.py

Generate matching files. The filename of each matching file represents the name of fingerprint. Each matching file contains matching scores with other fingerprints in a database. Specifically, each file takes this form "Comparison object    label     matching score", where label is 0 or 1. 0 or 1 represents whether fingerprint and comparison object correspond to the same finger or not. Thus, the performance indices such as EER, FMR100, ROC curves can be gotten according to these matching files.
## python match/generate_idy.py 

Calculate the matching time. We can count the runtime for n pairs of fingerprints in file "test/data/test_pairs.txt". Then average matching time equals the runtime divided by n.
## python match/predict.py
