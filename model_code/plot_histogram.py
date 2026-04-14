import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns
import os
import csv

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

from models.skip_alexnet import AlexNet
import torch

SMALLEST_SIZE = 18
SMALL_SIZE = 22
TITLE_SIZE = 20
MEDIUM_SIZE = 26
BIGGER_SIZE = 12

deg = u'\N{DEGREE SIGN}'

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALLEST_SIZE)    # legend fontsize
#plt.rc('title', fontsize=MEDIUM_SIZE)  # fontsize of the figure title

sns.set_palette("colorblind")

def read_data(dir, filename):
    file = open(dir+filename)
    reader = csv.reader(file)
    data = []
    for row in reader: data = row
    data = np.array(data, dtype=np.float64)
    return data

base_dir = 'saved_outputs/'
nonseq_dir = base_dir+'nonsequential/'
seq_dir = base_dir+'sequential/'
shuffled_dir = base_dir+'shuffled/'

stimulus_noise_sd = 0.02
confidence_noise_sd = 0.3

trials = range(1,21)

all_weights_nonseq = []
all_weights_seq = []
all_weights_shuff = []

nonseq_dir = nonseq_dir+'weights/'
seq_dir = seq_dir+'weights/'
shuffled_dir = shuffled_dir+'weights/'

for trial in trials:
    all_weights_nonseq.append(read_data(nonseq_dir,'final_fc_weights_ref_'+str(0)+'_sf_'+str(0.05)+'_sep_'+str(1.0)+'_lr_0.0001_trial_'+str(trial)+'.csv'))
    all_weights_seq.append(read_data(seq_dir,'final_fc_weights_ref_'+str(0)+'_sf_'+str(0.05)+'_sep_'+str(1.0)+'_lr_0.0001_trial_'+str(trial)+'.csv'))
    all_weights_shuff.append(read_data(shuffled_dir,'final_fc_weights_ref_'+str(0)+'_sf_'+str(0.05)+'_sep_'+str(1.0)+'_lr_0.0001_trial_'+str(trial)+'.csv'))

all_weights_nonseq = np.mean(np.array(all_weights_nonseq), axis=0)
all_weights_seq = np.mean(np.array(all_weights_seq), axis=0)
all_weights_shuff = np.mean(np.array(all_weights_shuff), axis=0)

weight_vectors = [all_weights_nonseq, all_weights_seq, all_weights_shuff]
labels = ['Nonseq', 'Seq', 'Shuff']

plt.figure(figsize=(5, 5), dpi=300)

for w, label in zip(weight_vectors, labels):
    abs_w = np.abs(w)

    # Choose bins that start at 0
    bins = np.linspace(0, np.max(abs_w), 51)

    # Histogram including zeros (now at the leftmost bin)
    hist_vals, bin_edges = np.histogram(abs_w, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Plot histogram
    plt.bar(bin_centers, hist_vals, width=bin_edges[1] - bin_edges[0], color=sns.color_palette("colorblind")[labels.index(label)],
            label=label)

plt.xlabel(r'$|\mathbf{w}|$')
plt.ylabel('Density')
plt.legend()
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
import matplotlib.ticker as ticker

def custom_formatter(x, pos):
    # Show 0 with no decimals; keep others with default formatting
    if x == 0:
        return '0'
    else:
        return f'{x:.3f}'  # Or adjust formatting as needed

ax = plt.gca()
ax.xaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))

plt.tight_layout()
save_dir = base_dir + 'plots/'
if not os.path.exists(save_dir): os.makedirs(save_dir)
plt.savefig(save_dir + 'histogram.svg')
# plt.show()
