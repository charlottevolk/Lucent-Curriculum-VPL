import matplotlib.pyplot as plt
import csv
import numpy as np
import scipy.stats as stats
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import starbars
import os

deg = u'\N{DEGREE SIGN}'

def read_data(dir, filename):
    file = open(dir+filename)
    reader = csv.reader(file)
    data = []
    for row in reader: data.append(row[0])
    data = np.array(data)
    data = data.astype(np.float64)
    return data

SMALLEST_SIZE = 12
SMALL_SIZE = 20
TITLE_SIZE = 20
MEDIUM_SIZE = 30
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALLEST_SIZE)    # legend fontsize
#plt.rc('title', fontsize=MEDIUM_SIZE)  # fontsize of the figure title

sns.set_palette("colorblind", 3)

base_dir = 'saved_outputs/'
save_dir = 'saved_outputs/plots/'

trials = range(1,21)
noise_sd = 0.02
confidence_sd = 0.3

model_labels = ['Non-sequential', 'Shuffled', 'Sequential']
model_dirs = [base_dir + 'skip_nonseq_5.0_1.0_noise_sd_'+str(noise_sd)+'_added_confidence_noise_sd_'+str(confidence_sd)+'_single_sample_update/',
              base_dir + 'skip_shuff_5.0_1.0_noise_sd_'+str(noise_sd)+'_added_confidence_noise_sd_'+str(confidence_sd)+'_single_sample_update/',
              base_dir + 'skip_seq_5.0_1.0_noise_sd_'+str(noise_sd)+'_added_confidence_noise_sd_'+str(confidence_sd)+'_single_sample_update/',
            ]

fig, ax = plt.subplots(figsize=(7,5), dpi=300)
plt.xlabel("N", labelpad=12)
plt.ylabel(r"$\Delta$Acc./N", labelpad=12)
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

num_neurons = 250
neurons_lesioned = list(range(10, num_neurons+1, 10))

for model_dir, model_name in zip(model_dirs, model_labels):
    transfer_accuracies = []
    for trial_num in trials:
        base_acc = np.mean(read_data(model_dir, 'data/skip/SF_doubled_lr_0.0001/0_neurons_transfer_accuracy_ref_0_sf_0.1_sep_1.0_lr_0.0001_trial_'+str(trial_num)+'.csv'))
        t_acc_trial = []
        for neurons in neurons_lesioned:
            t_acc = read_data(model_dir, 'data/skip/SF_doubled_lr_0.0001/'+str(neurons)+'_neurons_transfer_accuracy_ref_0_sf_0.1_sep_1.0_lr_0.0001_trial_'+str(trial_num)+'.csv')
            t_acc_trial.append((base_acc - np.mean(t_acc))/neurons)
        transfer_accuracies.append(t_acc_trial)
    acc_mean = np.mean(transfer_accuracies, axis=0)
    acc_std = np.std(transfer_accuracies, axis=0)

    plt.plot(neurons_lesioned, acc_mean, lw=5, label=model_name, color=sns.color_palette("colorblind")[model_labels.index(model_name)])
    plt.scatter(neurons_lesioned, acc_mean, s=150, color=sns.color_palette("colorblind")[model_labels.index(model_name)])
    plt.fill_between(neurons_lesioned, acc_mean-acc_std, acc_mean+acc_std, alpha=0.3, color=sns.color_palette("colorblind")[model_labels.index(model_name)])

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend()
plt.tight_layout()
plt.savefig(save_dir+'neuron_lesioning_plot.svg', dpi=300)