import sys
from preamble import *
from fed import main
import sample
import json

# measure by AK if true distribution is unstructured
if STRUCTURED:
	def error_measure(p1,p2,sparsity=SPARSITY):
		return err(p1,p2)
# measure by TV otherwise
else:
	def error_measure(p1,p2,sparsity=SPARSITY):
		return AK(sparsity,p1,p2)

np.random.seed(SEED)

# runs first experiment, i.e. vary domain size
def experiment_1():
	eps = 0.4
	k = 1000
	N = int((SPARSITY/eps**2)/(1.-eps))
	ns = [4,8,16,32,64,128]
	
	our_errs = np.zeros((len(ns),NUM_TRIALS))
	clean_errs = np.zeros((len(ns),NUM_TRIALS))
	emp_errs = np.zeros((len(ns),NUM_TRIALS))
	for t in range(NUM_TRIALS):
		for i,n in enumerate(ns):
			worst_phat_err = -np.inf
			worst_phat_err_id = None
			p, qs = sample.get_dists(n)
			full_data, clean_mean, emp_means = sample.get_batches(p,qs,k,eps,N)
			# pick the corruption that makes our estimator do worst
			for j in range(NUM_CORRUPTIONS):
				data = full_data[j,:,:]
				phat = main(data,k,SPARSITY,p)
				if STRUCTURED:
					phat = AKround(phat,SPARSITY)
				phat_err = error_measure(phat, p)
				if phat_err > worst_phat_err:
					worst_phat_err = phat_err
					worst_phat_err_id = j

			our_err = worst_phat_err
			clean_err = error_measure(clean_mean, p)
			emp_err = error_measure(emp_means[worst_phat_err_id,:], p)

			our_errs[i,t] = our_err
			clean_errs[i,t] = clean_err
			emp_errs[i,t] = emp_err
	if STRUCTURED:
		filename = 'data/structured_exp_1-seed%d.json' % SEED
	else:
		filename = 'data/exp_1-seed%d.json' % SEED
	with open(filename,'w') as f:
		json.dump([our_errs.tolist(), clean_errs.tolist(), emp_errs.tolist()],f)

# runs second experiment, i.e. vary k
def experiment_2():
	eps = 0.4
	n = 64
	ks = [1, 50, 100, 250, 500, 750, 1000]

	N = int((SPARSITY/eps**2)/(1.-eps))
	
	our_errs = np.zeros((len(ks),NUM_TRIALS))
	clean_errs = np.zeros((len(ks),NUM_TRIALS))
	emp_errs = np.zeros((len(ks),NUM_TRIALS))
	for t in range(NUM_TRIALS):
		p, qs = sample.get_dists(n)
		for i,k in enumerate(ks):
			worst_phat_err = -np.inf
			worst_phat_err_id = None
			full_data, clean_mean, emp_means = sample.get_batches(p,qs,k,eps,N)
			for j in range(NUM_CORRUPTIONS):
				data = full_data[j,:,:]
				phat = main(data,k,SPARSITY,p)
				if STRUCTURED:
					phat = AKround(phat,SPARSITY)
				phat_err = error_measure(phat, p)
				if phat_err > worst_phat_err:
					worst_phat_err = phat_err
					worst_phat_err_id = j

			our_err = worst_phat_err
			clean_err = error_measure(clean_mean, p)
			emp_err = error_measure(emp_means[worst_phat_err_id,:], p)

			our_errs[i,t] = our_err
			clean_errs[i,t] = clean_err
			emp_errs[i,t] = emp_err
	if STRUCTURED:
		filename = 'data/structured_exp_2-seed%d.json' % SEED
	else:
		filename = 'data/exp_2-seed%d.json' % SEED
	with open(filename,'w') as f:
		json.dump([our_errs.tolist(), clean_errs.tolist(), emp_errs.tolist()],f)

# runs third experiment, i.e. vary eps
def experiment_3():
	n = 64
	epss = [0.0, 0.1, 0.2, 0.3, 0.4, 0.49]
	k = 1000
	
	our_errs = np.zeros((len(epss),NUM_TRIALS))
	clean_errs = np.zeros((len(epss),NUM_TRIALS))
	emp_errs = np.zeros((len(epss),NUM_TRIALS))
	N = int(SPARSITY/max(epss)**2)
	
	for t in range(NUM_TRIALS):
		p,qs = sample.get_dists(n)
		clean_data, clean_mean = sample.get_good_batches(p,k,N)
		for i, eps in enumerate(epss):
			worst_phat_err = -np.inf
			worst_phat_err_id = None

			dirty_data = sample.get_bad_batches(qs,k,int(N*eps/(1-eps)))
			full_data, emp_means = sample.merge_data(clean_data, dirty_data)
			# full_data, clean_mean, emp_means = sample.get_batches(p,qs,k,eps,N)

			for j in range(NUM_CORRUPTIONS):
				data = full_data[j,:,:]
				phat = main(data,k,SPARSITY,p)
				if STRUCTURED:
					phat = AKround(phat,SPARSITY)
				phat_err = error_measure(phat, p)
				if phat_err > worst_phat_err:
					worst_phat_err = phat_err
					worst_phat_err_id = j

			our_err = worst_phat_err
			clean_err = error_measure(clean_mean, p)
			emp_err = error_measure(emp_means[worst_phat_err_id,:], p)

			print "TRIAL #", t
			print eps, our_err, clean_err, emp_err

			our_errs[i,t] = our_err
			clean_errs[i,t] = clean_err
			emp_errs[i,t] = emp_err
	if STRUCTURED:
		filename = 'data/structured_exp_3-seed%d.json' % SEED
	else:
		filename = 'data/exp_3-seed%d.json' % SEED
	with open(filename,'w') as f:
		json.dump([our_errs.tolist(), clean_errs.tolist(), emp_errs.tolist()],f)

# runs fourth experiment, i.e. vary N
def experiment_4():
	# optionally, can tweak this to run experiment
	# for different choices of \ell, but we just
	# try a single choice of \ell in the paper
	sparsities = [SPARSITY]
	eps = 0.4
	n = 128
	ratios = [0.75,1.,1.25,1.5,1.75]

	k = 500
	our_errs = [0]*len(sparsities)
	clean_errs = [0]*len(sparsities)
	emp_errs = [0]*len(sparsities)
	sample_complexities = [0]*len(sparsities)
	
	for sparsity_i, sparsity in enumerate(sparsities):
		max_N = (sparsity/eps**2)
		Ns = [int(r*max_N) for r in ratios]
		sample_complexities[sparsity_i] = Ns
		this_our_errs = np.zeros((len(Ns),NUM_TRIALS))
		this_clean_errs = np.zeros((len(Ns),NUM_TRIALS))
		this_emp_errs = np.zeros((len(Ns),NUM_TRIALS))
		for t in range(NUM_TRIALS):
			p, qs = sample.get_dists(n)
			for i, N in enumerate(Ns):
				worst_phat_err = -np.inf
				worst_phat_err_id = 0
				full_data, clean_mean, emp_means = sample.get_batches(p,qs,k,eps,N)
				
				for j in range(NUM_CORRUPTIONS):
					data = full_data[j,:,:]
					phat = main(data,k,sparsity,p)
					if STRUCTURED:
						phat = AKround(phat,SPARSITY)
					phat_err = error_measure(phat, p)
					if phat_err > worst_phat_err:
						worst_phat_err = phat_err
						worst_phat_err_id = j

				our_err = worst_phat_err
				clean_err = error_measure(clean_mean, p, sparsity=sparsity)
				emp_err = error_measure(emp_means[worst_phat_err_id,:], p, sparsity=sparsity)

				print "TRIAL:", t
				print "sparsity:", sparsity
				print "#samples:", N
				print our_err, clean_err, emp_err, eps/np.sqrt(k)

				this_our_errs[i,t] = our_err
				this_clean_errs[i,t] = clean_err
				this_emp_errs[i,t] = emp_err
	
		our_errs[sparsity_i] = this_our_errs.tolist()
		clean_errs[sparsity_i] = this_clean_errs.tolist()
		emp_errs[sparsity_i] = this_emp_errs.tolist()

	if STRUCTURED:
		filename = 'data/structured_exp_4-seed%d.json' % SEED
	else:
		filename = 'data/exp_4-seed%d.json' % SEED
	with open(filename,'w') as f:
		json.dump([our_errs, clean_errs, emp_errs, sample_complexities],f)


if sys.argv[1]=='1':
	experiment_1()
elif sys.argv[1]=='2':
	experiment_2()
elif sys.argv[1]=='3':
	experiment_3()
elif sys.argv[1]=='4':
	experiment_4()
else:
	print "invalid argument"
