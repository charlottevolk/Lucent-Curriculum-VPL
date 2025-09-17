import matplotlib.pyplot as plt
import csv
import numpy as np
import scipy.stats as stats
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import starbars
import os
import torch
from scipy.stats import linregress, sem

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

stimulus_noise_sd = 0.02
confidence_noise_sd = 0.3

num = 20

base_dir = 'saved_outputs/'
save_dir = 'saved_outputs/plots/'

trials = range(1,21)

transfer_accuracies = []
y_errors = []
x_vals = []

plt.figure(figsize=(5,5), dpi=300)

num_steps = 25
factors_seq = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
factors_antiseq = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0]
factor_x_axis = range(-9, 10)
tick_positions = [-9, 0, 9]  # Corresponds to factors of 1:10, 1:5, 1:1, 5:1, 10:1
tick_labels = ['1:10', '1:1', '10:1']

for factor in factors_antiseq:
    transfer_data = np.array([read_data(base_dir+'skip_shuff_5.0_1.0_noise_sd_'+str(stimulus_noise_sd)+'_added_confidence_noise_sd_'+str(confidence_noise_sd)+'_single_sample_update_forced_antisequential_with_weighted_samples_beginning_'+str(num_steps)+'_steps_factor_'+str(factor)+'/', 'data/skip/SF_doubled_lr_0.0001/transfer_accuracy_ref_'+str(0)+'_sf_'+str(0.1)+'_sep_'+str(1.0)+'_lr_0.0001_trial_'+str(trial)+'.csv') for trial in trials])
    transfer_acc = np.mean(transfer_data)
    transfer_data = np.mean(transfer_data, axis=1)
    acc_sem = sem(transfer_data)
    transfer_accuracies.append(transfer_acc)
    y_errors.append(acc_sem)
    x_vals.append(-factor+1)

for factor in factors_seq:
    transfer_data = np.array([read_data(base_dir+'skip_shuff_5.0_1.0_noise_sd_'+str(stimulus_noise_sd)+'_added_confidence_noise_sd_'+str(confidence_noise_sd)+'_single_sample_update_forced_sequential_with_weighted_samples_beginning_'+str(num_steps)+'_steps_factor_'+str(factor)+'/', 'data/skip/SF_doubled_lr_0.0001/transfer_accuracy_ref_'+str(0)+'_sf_'+str(0.1)+'_sep_'+str(1.0)+'_lr_0.0001_trial_'+str(trial)+'.csv') for trial in trials])
    transfer_acc = np.mean(transfer_data)
    transfer_data = np.mean(transfer_data, axis=1)
    acc_sem = sem(transfer_data)
    transfer_accuracies.append(transfer_acc)
    y_errors.append(acc_sem)
    x_vals.append(factor-1)

factor_x_axis = np.array(factor_x_axis)
x_vals = np.array(x_vals)
transfer_accuracies = np.array(transfer_accuracies)

sns.regplot(x=x_vals, y=transfer_accuracies, scatter=False, ci=95, color='black', line_kws={'linestyle': '--'})

slope, intercept, r_value, p_value, _ = linregress(x_vals, transfer_accuracies)

plt.plot([], [], label=f'$r$ = {r_value:.2f}, p = {p_value:.2g}', color='black', linestyle='--')

plt.errorbar(x_vals, transfer_accuracies, yerr=y_errors, fmt='o', capsize=3, color='tab:blue')
plt.scatter(x_vals, transfer_accuracies, s=150, color='tab:blue')

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xticks(ticks=tick_positions, labels=tick_labels)
plt.yticks(ticks=[0.85, 0.9])
plt.xlabel('Easy:hard samples', labelpad=12)
plt.ylabel('Transfer accuracy', labelpad=12)
plt.tight_layout()
plt.legend()

plt.savefig(save_dir+'accuracy_correlation_forced_curriculum.svg')

