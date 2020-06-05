import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import json
import seaborn as sns
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FormatStrFormatter

# one color per line in plot
NUM_COLORS = 4
# list of \ell's that we tried
sparsities = [5]
# avg is the way we aggregate results to make plots
avg = np.median
# choice of random seed used, for bookkeeping here
SEED = 0

# if True, use 1 stdev for error bars (we do this in the paper)
USE_STDEV = True
# if False, we will use TOP/BOTTOM-th quantiles
TOP = 0.75
BOTTOM = 0.25

# if plot_structured is True, plots experiment (B) results, else (A)
def compact_plot(plot_structured,linewidth=1.5,markersize=2):
	def filename(seed, exp_number):
		if plot_structured:
			filename = 'data/structured_exp_%d-seed%d.json' % (exp_number,seed)
		else:
			filename = 'data/exp_%d-seed%d.json' % (exp_number,seed)
		return filename
	fig, axs = plt.subplots(nrows=2, ncols=2)
	sns.set_style("ticks", {'grid.linestyle': '--'})
	sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
	cm = plt.get_cmap('plasma')
	fontP = FontProperties()
	fontP.set_size('xx-small')

	for exp_number in range(1,5):
		print (exp_number - 1) / 2
		print (exp_number - 1) % 2
		ax = axs[(exp_number-1) / 2][(exp_number-1) % 2]
		ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
		ax.grid(color='dimgray',linestyle='--',linewidth=0.25)
		ax.set_prop_cycle(color = [cm(1.3*i/NUM_COLORS) for i in range(NUM_COLORS)])

		if plot_structured:
			ax.set_ylabel('$L_1$ distance')
		else:
			ax.set_ylabel('$A_{\ell/2}$ distance')
		if exp_number == 1:
			xs = [4,8,16,32,64,128]
			ax.set_xlabel('(i) domain size $n$')
		elif exp_number == 2:
			xs = [1,50, 100, 250, 500, 750, 1000]
			ax.set_xlabel('(ii) batch size $k$')
		elif exp_number == 3:
			xs = [0.0, 0.1, 0.2, 0.3, 0.4]
			ax.set_xlabel('(iii) corruption $\epsilon$')
		elif exp_number == 4:
			max_N = sparsities[0]/0.4**2
			ratios = [0.75,1.,1.25,1.5,1.75]
			xs = [int(r*max_N) for r in ratios]
			ax.set_xlabel('(iv) number of batches')
		else:
			raise("experiment not supported")

		for _,s in ax.spines.items():
			s.set_linewidth(0.8)

		with open(filename(SEED,exp_number), 'r') as f:
			data = np.array(json.load(f))

		if exp_number in [1,2,3]:
			our_errs = data[0,:,:]
			clean_errs = data[1,:,:]
			emp_errs = data[2,:,:]
		else:
			our_errs = np.array([np.array(x) for x in data[0][0]])
			clean_errs = np.array([np.array(x) for x in data[1][0]])
			emp_errs = np.array([np.array(x) for x in data[2][0]])

		if USE_STDEV:
			our_error_bars = np.std(our_errs,axis=1)
			clean_error_bars = np.std(clean_errs,axis=1)
			emp_error_bars = np.std(emp_errs,axis=1)
		else:
			our_error_bars = np.quantile(our_errs,TOP,axis=1) - np.quantile(our_errs,BOTTOM,axis=1)
			clean_error_bars = np.quantile(clean_errs,TOP,axis=1) - np.quantile(clean_errs,BOTTOM,axis=1)
			emp_error_bars = np.quantile(emp_errs,TOP,axis=1) - np.quantile(emp_errs,BOTTOM,axis=1)
		
		ax.errorbar(xs,avg(our_errs,axis=1)[:len(xs)],yerr=our_error_bars[:len(xs)],marker='.',ms=markersize,label='filter',linewidth=linewidth)
		ax.errorbar(xs,avg(clean_errs,axis=1)[:len(xs)],yerr=clean_error_bars[:len(xs)],marker='.',ms=markersize,label='oracle',linewidth=linewidth)
		ax.errorbar(xs,avg(emp_errs,axis=1)[:len(xs)],yerr=emp_error_bars[:len(xs)],marker='.',ms=markersize,label='naive',linewidth=linewidth)
		mesh = np.linspace(min(xs),max(xs),100)
		if exp_number == 1:
			ax.plot(xs,[0.4/np.sqrt(1000)]*len(xs),linestyle='dotted',linewidth=1.,color='green',label='$\epsilon/\sqrt{k}$')
		elif exp_number == 2:
			ax.plot(mesh,[0.4/np.sqrt(k) for k in mesh],linestyle='dotted',linewidth=1.,color='green',label='$\epsilon/\sqrt{k}$')
		elif exp_number == 3:
			ax.plot(xs,[0.4/np.sqrt(1000)]*len(xs),linestyle='dotted',linewidth=1.,color='green',label='$\epsilon/\sqrt{k}$')
		elif exp_number == 4:
			ax.plot(xs,[0.4/np.sqrt(500)]*len(xs),linestyle='dotted',linewidth=1.,color='green',label='$\epsilon/\sqrt{k}$')		
		else:
			print exp_number
		ax.legend(loc="right",prop={'size': 6})
	if plot_structured:
		title = 'main_structured'
	else:
		title = 'main_unstructured'
	fig.tight_layout(pad=0.5)
	plt.savefig('figures/%s.pdf' % title, transparent="True",dpi=200)

compact_plot(True)
compact_plot(False)
