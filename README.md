# Code for Learning Structured Distributions From Untrusted Batches: Faster and Simple"

## Contents
- `preamble.py`: contains main dependencies and global flags
- `fed.py`: implementation of our algorithm
- `sample.py`: code for generating synthetic data from untrusted batches
- `experiments.py`: wrapper code to run all of our experiments
- `data/` is pre-populated with our experimental data.
- To reproduce the plots in this work, run `python plots.py`
- To re-generate data for experiment number i from scratch, run `python experiments.py i` for any choice of i = 1,2,3,4. See paper for estimated runtimes.
