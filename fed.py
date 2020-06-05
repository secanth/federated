from preamble import *
import json

# l\in[n]
# computes mu_{i,j}, where (i,j) is the index in the Haar wavelet basis in 2^m dimensions
def mu(m,l):
    if l == 0 or l == 1:
        return 2**(-m/2.)
    else:
        i = int(np.log(l)/np.log(2))
        return 2**(-(m-i)/2.)

# generates the Haar basis matrix weighted to be 0/1
def gen_Y(n):
    m = int(np.log(n)/np.log(2))
    H = np.zeros((n,n))
    H[0,:] = n**(-1/2.) * np.ones(n)
    H[1,:] = n**(-1/2.) * np.array([1.]*(n/2) + [-1.]*(n/2))
    counter = 2
    for i in range(1,m):
        for j in range(2**i):
            length = 2**(m-i-1)
            start = j*(2**(m-i))
            mid = start + length
            end = start + 2*length
            H[counter,start:mid] = np.ones(length) * (2 ** (-(m-i)/2.))
            H[counter,mid:end] = -np.ones(length)*(2 ** (-(m-i)/2.))
            counter += 1
    Dinv = np.diag([1./mu(m,l) for l in range(n)])
    Y = np.matmul(H.T,Dinv).astype(int)
    return Y

# solves SDP sup_{\Sigma\in K}|<X,Sigma>| for \ell = sparsity
# input Y should be the Haar basis matrix weighted to be 0/1
def sdp(X,Y,sparsity):
    signchanges = 2*sparsity
    n,_ = X.shape
    A = cp.Variable((n,n))
    B = cp.Variable((n,n), PSD = True)
    constraints = []
    constraints.append(cp.norm(A,'fro') <= np.sqrt(signchanges*np.log(n) + 1))
    constraints.append(cp.pnorm(A,1) <= signchanges*np.log(n) + 1)
    constraints.append(cp.pnorm(A,'inf') <= 1.)
    constraints.append(cp.pnorm(B, 'inf') <= 1.)
    constraints.append(B == Y * A * Y.T)
    obj1 = cp.Maximize(cp.trace(B * X))
    prob1 = cp.Problem(obj1, constraints)
    obj2 = cp.Minimize(cp.trace(B * X))
    prob2 = cp.Problem(obj2, constraints)
    # need to set feasibility tolerance quite high...
    result1 = prob1.solve(eps=1e-2)
    B1 = B.value
    result2 = prob2.solve(eps=1e-2)
    B2 = B.value
    if np.abs(result1) > np.abs(result2):
        return B1, np.abs(result1)
    else:
        return B2, np.abs(result2)

# returns vector given by w-weighted mean of points
def mean(w,points):
    total = np.sum((w * points.T).T,axis=0)
    total /= np.sum(total)
    return total

# returns mask of which points still in the support of w
def live(w):
    return w > 0.

# updates w using scores (1DFilter)
def update_weights(w,scores):
    live_points = live(w)
    taumax = float(np.max(scores[live_points]))
    # failure mode if SDP solver doesn't do well:
    if taumax < 0:
        raise('PSDness violated')
    # updates the array of weights
    w[live_points] *= (1. - scores[live_points]/taumax)

# run the main algorithm
# NOTE: p, the true distribution, is just an input for debugging purposes
# sparsity is the K in AK norm, so 2*sparsity is number of sign changes
def main(data,k,sparsity,p):
    N,n = data.shape
    Y = gen_Y(n)
    prev_skewness = np.inf
    w = np.ones(N)/float(N)
    old_muw = None
    while True:
        # get current weighted mean
        muw = mean(w,data)
        # (for debugging purposes, compute current AK error)
        AKdistance = AK(sparsity,muw,p)

        # recenter data around weighted mean
        centered = data - muw

        # compute M(w)
        Aw = np.matmul(centered.T, np.matmul(np.diag(w), centered))
        Bw = 1./k * (np.diag(muw) - np.outer(muw,muw))
        Mw = Aw - Bw

        # compute current skewness, and maximizing Sigma
        Sigma, skewness = sdp(Mw,Y,sparsity)

        # if skewness has gone back up, break, output old mean
        if skewness > prev_skewness:
            return old_muw

        prev_skewness = skewness

        # compute scores based on Sigma
        scores = np.diag(np.matmul(centered,np.matmul(Sigma,centered.T)))
        # run 1DFilter
        try:
            update_weights(w,scores)
            old_muw = muw
        # because of the high feasibility tolerance for the SDP solver
        # there may be an issue with updating weights because a score
        # might be negative, in which case we just output the current mean
        except:
            print "PSDness failure"
            return muw
