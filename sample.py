from preamble import *

# sample a random num_pieces-wise constant distribution over [n]
def piecewise(n,num_pieces):
	out = np.zeros(n)
	ids = np.sort(np.random.choice(range(n),num_pieces-1,replace=False))
	ids = np.hstack(([0],ids,[n-1]))
	for i in range(num_pieces):
		print ids[i], ids[i+1]
		out[ids[i]:ids[i+1]] = np.random.random()
	return out/np.sum(out)

# iid Unif([0,1]) marginals + normalization
def rando(n):
	out = np.random.random(n)
	return out / np.sum(out)

# corrupt p by a prescribed TV distance
def corrupt(p,distance):
	n = len(p)
	ranks = np.argsort(p)
	memo = {ranks[i]:i for i in range(n)}
	q = np.sort(p)
	for i in range(n/2):
		q[i] += distance/(n/2)
		q[n-1-i] -= distance/(n/2)
	if np.sum((q < 0) | (q > 1)) == 0:
		out = [0]*n
		for i in range(n):
			out[i] = q[memo[i]]
		return out / np.sum(out)
	else:
		raise("q generation failed")

# samples N points from Mul(k,p)
def sample_mult(p,k,N):
	out = multinomial(k,p).rvs(N).astype(float)
	return out / float(k)

# obtain true distribution p as well as
# qs consisting of corruptions of p at 
# NUM_CORRUPTIONS different corruption distances
# for the paper, we just took NUM_CORRUPTIONS = 1
def get_dists(n):
	qs = [0]*NUM_CORRUPTIONS
	while True:
		if STRUCTURED:
			p = piecewise(n,SPARSITY)
		else:
			p = rando(n)
		q_fail = False
		counter = 0
		for distance in np.linspace(MIN_CORRUPTION,0.6,NUM_CORRUPTIONS):
			try:
				q = corrupt(p,distance)
				qs[counter] = q
			except:
				q_fail = True
				break
			counter += 1
		if not q_fail:
			break
	return p, qs

# draw N1 good batches, i.e. iid draws from Mul(k,p)
def get_good_batches(p,k,N1):
	n = len(p)
	clean_data = sample_mult(p,k,N1)
	clean_mean = np.average(clean_data,axis=0)
	return clean_data, clean_mean

# draw N2 good batches, i.e. iid draws from Mul(k,q)
# for each q among the possible corruptions qs
def get_bad_batches(qs,k,N2):
	n = len(qs[0])
	dirty_data = np.zeros((NUM_CORRUPTIONS,N2,n))
	for counter, q in enumerate(qs):
		dirty = sample_mult(q,k,N2)
		dirty_data[counter,:,:] = dirty
	return dirty_data

# concatenate good and bad points
def merge_data(clean_data, dirty_data):
	N1,n = clean_data.shape
	_,N2,_ = dirty_data.shape
	N = N1 + N2
	# print N
	full_data = np.zeros((NUM_CORRUPTIONS,N,n))
	emp_means = np.zeros((NUM_CORRUPTIONS,n))

	for counter in range(NUM_CORRUPTIONS):
		data = np.vstack((clean_data,dirty_data[counter,:,:]))
		full_data[counter,:,:] = data
		emp_means[counter,:] = np.average(data,axis=0)
	return full_data, emp_means

# wrapper function for getting data
def get_batches(p,qs,k,eps,N):
	good = int(N*(1 - eps))
	bad = N - good
	clean_data, clean_mean = get_good_batches(p,k,good)
	dirty_data = get_bad_batches(qs,k,bad)
	full_data, emp_means = merge_data(clean_data,dirty_data)
	return full_data, clean_mean, emp_means
