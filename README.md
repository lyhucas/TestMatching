# TestMatching
I organized this reposity mainly for testing our fingerprint matching algorithm

# Requirements 
tensorflow 1.x
Python 3.x

# Usage
## extract fingerprint features
python match/extract_feature.py

## generate matching files. The filename of each matching file represents the name of fingerprint. Each matching file contains matching scores with other fingerprints in a database. Specifically, each file takes this form "Comparison object    label     matching score", where label is 0 or 1. 0 or 1 represents 

python match/generate_idy.py 

python
