from scipy.stats import multinomial
import cvxpy as cp
import numpy as np
import itertools
import matplotlib.pyplot as plt

# number of possible corruptions to try for a given p
# for this paper, we just try 1, simply for convenience
NUM_CORRUPTIONS = 1
# number of trials per data point
NUM_TRIALS = 10
# SPARSITY = \ell/2, i.e. number of pieces of 
# piecewise constant p that we'd estimate
SPARSITY = 5
# True if running experiment (B), False otherwise
STRUCTURED = True
# random seed for reproducibility purposes
SEED = 0

# distance by which we perturb true distribution
# see paper for explanation for why we chose these
if STRUCTURED:
    MIN_CORRUPTION = 0.3
else:
    MIN_CORRUPTION = 0.5

# TV distance between p1 and p2, i.e. half of L1
def err(p1,p2):
    return np.sum(np.abs(p1-p2))/2.

# computes AK distance between distributions p1 and p2 (value between 0 and 1)
def AK(K,p1,p2):
    diff = p1 - p2
    n = len(diff)
    memo = {}
    best_vecs = {}
    # solves subproblem starting at index i, with l allowed sign changes, with first sign equal to s
    def helper(l,i,s):
        if (l,i,s) in memo:
            return memo[(l,i,s)], best_vecs[(l,i,s)]
        elif i >= n:
            return 0, np.array([])
        elif l == 0:
            return s*np.sum(diff[i:]), s*np.ones(n-i)
        else:
            best_val = -np.inf
            best_vec = None
            for j in range(i,n):
                pre_val,pre_vec = helper(l-1,j,-s) 
                val = pre_val + s*np.sum(diff[i:j])
                if val > best_val:
                    best_val = val
                    best_vec = np.concatenate((s*np.ones(j - i),pre_vec))
            memo[(l,i,s)] = best_val
            best_vecs[(l,i,s)] = best_vec
            return best_val, best_vec
    val1, vec1 = helper(K,0,1)
    val2, vec2 = helper(K,0,-1)
    if val1 > val2:
        return val1/2.
    else:
        return val2/2.

# replaces a list of numbers with all-ones times its average
def basic_flatten(l):
    out = np.ones(len(l))*np.average(l)
    return out

# flattens list (in the sense of basic_flatten) at the 
# specified breakpoints
def flattened(l,breakpoints):
    num_points = len(breakpoints)
    out = [0]*len(l)
    for i in range(num_points - 1):
        out[breakpoints[i]:breakpoints[i+1]] = basic_flatten(l[breakpoints[i]:breakpoints[i+1]])
    return out

# computes AK distance between list l and its flattening:
def flat_err(l):
    return AK(1,l,basic_flatten(l))

# rounds p to closest K-wise constant distribution
def AKround(p,K):
    m = 1
    n = len(p)
    breakpoints = range(n+1)
    while True:
        num_points = len(breakpoints)
        if num_points - 1 <= 2*K:
            return flattened(p,breakpoints)
        num_pairs = num_points - 2
        errs = np.zeros(num_pairs)
        for i in range(num_pairs):
            l1 = basic_flatten(p[breakpoints[i]:breakpoints[i+1]])
            l2 = basic_flatten(p[breakpoints[i+1]:breakpoints[i+2]])
            errs[i] = flat_err(np.hstack((l1,l2)))
        top_ids = (errs).argsort()[:m]
        breakpoints = np.delete(breakpoints,[i+1 for i in top_ids])
