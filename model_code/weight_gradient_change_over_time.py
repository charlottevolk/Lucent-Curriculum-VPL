import matplotlib.pyplot as plt
import csv
import numpy as np
from statistics import mean
import torch
import pandas as pd
import seaborn as sns
import matplotlib.colors as mcolors
from scipy.stats import sem

SMALLEST_SIZE = 18
SMALL_SIZE = 22
TITLE_SIZE = 20
MEDIUM_SIZE = 28
BIGGER_SIZE = 12

base_dir = 'saved_outputs/'
single_angle_sep_dir = base_dir + 'single_angle_sep_doubled_SF//'

weight_dir = single_angle_sep_dir + '0.5_angle_sep/weights/'
save_dir = base_dir + 'plots/'

iterations = np.arange(1,501)

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALLEST_SIZE)    # legend fontsize
#plt.rc('title', fontsize=MEDIUM_SIZE)  # fontsize of the figure title

sns.set_palette('GnBu_r',5)

deg = u'\N{DEGREE SIGN}'

# Train plot

#[ref ori, sf, sep]
train_params = [
            [0,0.05,0.5]
            ]

num_trials = 20
trials = np.arange(1,num_trials+1)

conv_all_weights = []

# Skip connection weights

for run in train_params:

    all_c1_weights = []
    all_c2_weights = []
    all_c3_weights = []
    all_c4_weights = []
    all_c5_weights = []

    full_grads = []
    plt.figure(figsize=(5,5), dpi=300)
    
    for trial in trials:

        ref_ori = run[0]
        sf = run[1]
        sep = run[2]

        np_data = np.load(weight_dir+'all_fc_weights_ref_'+str(ref_ori)+'_sf_'+str(sf)+'_sep_'+str(sep)+'_lr_0.0001_trial_'+str(trial)+'.npy', allow_pickle=True)

        layer_sizes = [64*56*56,
                    64*27*27,
                    192*27*27,
                    192*13*13,
                    384*13*13,
                    256*13*13,
                    256*13*13,
                    256*6*6
                    ]

        curr_index = 0
        c1_weights = []
        c2_weights = []
        c3_weights = []
        c4_weights = []
        c5_weights = []
        for row in np_data:
            weights = []
            curr_index = 0
            for i in range(0,len(layer_sizes)):
                weights.append(row[0][curr_index:curr_index+layer_sizes[i]])
                curr_index += layer_sizes[i]

            weights = np.array(weights, dtype=object)

            c1 = weights[0]
            c2 = weights[2]
            c3 = weights[4]
            c4 = weights[5]
            c5 = weights[6]

            c1_weights.append(np.linalg.norm(c1))
            c2_weights.append(np.linalg.norm(c2))
            c3_weights.append(np.linalg.norm(c3))
            c4_weights.append(np.linalg.norm(c4))
            c5_weights.append(np.linalg.norm(c5))

        all_c1_weights.append(c1_weights)
        all_c2_weights.append(c2_weights)
        all_c3_weights.append(c3_weights)
        all_c4_weights.append(c4_weights)
        all_c5_weights.append(c5_weights)
            
    # Compute absolute gradients (rate of change) for skip connections
    skip_c1_rate = np.array([np.abs(np.gradient(np.array(c1_weights))) for c1_weights in all_c1_weights])
    skip_c2_rate = np.array([np.abs(np.gradient(np.array(c2_weights))) for c2_weights in all_c2_weights])
    skip_c3_rate = np.array([np.abs(np.gradient(np.array(c3_weights))) for c3_weights in all_c3_weights])
    skip_c4_rate = np.array([np.abs(np.gradient(np.array(c4_weights))) for c4_weights in all_c4_weights])
    skip_c5_rate = np.array([np.abs(np.gradient(np.array(c5_weights))) for c5_weights in all_c5_weights])
    
    skip_c1_mean = np.mean(skip_c1_rate, axis=0)
    skip_c2_mean = np.mean(skip_c2_rate, axis=0)
    skip_c3_mean = np.mean(skip_c3_rate, axis=0)
    skip_c4_mean = np.mean(skip_c4_rate, axis=0)
    skip_c5_mean = np.mean(skip_c5_rate, axis=0)
    
    skip_c1_std = sem(skip_c1_rate, axis=0)
    skip_c2_std = sem(skip_c2_rate, axis=0)
    skip_c3_std = sem(skip_c3_rate, axis=0)
    skip_c4_std = sem(skip_c4_rate, axis=0)
    skip_c5_std = sem(skip_c5_rate, axis=0)

    x_vals = np.arange(1,501,20)

    plt.plot(x_vals, skip_c1_mean, label='Conv1', lw=5, solid_capstyle='round', zorder=10)
    plt.plot(x_vals, skip_c2_mean, label='Conv2', lw=5, solid_capstyle='round', zorder=9)
    plt.plot(x_vals, skip_c3_mean, label='Conv3', lw=5, solid_capstyle='round', zorder=8)
    plt.plot(x_vals, skip_c4_mean, label='Conv4', lw=5, solid_capstyle='round', zorder=7)
    plt.plot(x_vals, skip_c5_mean, label='Conv5', lw=5, solid_capstyle='round', zorder=6)

    plt.fill_between(x_vals, skip_c1_mean-skip_c1_std, skip_c1_mean+skip_c1_std, alpha=0.3, zorder=10)
    plt.fill_between(x_vals, skip_c2_mean-skip_c2_std, skip_c2_mean+skip_c2_std, alpha=0.3, zorder=9)
    plt.fill_between(x_vals, skip_c3_mean-skip_c3_std, skip_c3_mean+skip_c3_std, alpha=0.3, zorder=8)
    plt.fill_between(x_vals, skip_c4_mean-skip_c4_std, skip_c4_mean+skip_c4_std, alpha=0.3, zorder=7)
    plt.fill_between(x_vals, skip_c5_mean-skip_c5_std, skip_c5_mean+skip_c5_std, alpha=0.3, zorder=6)

    plt.xlabel("Steps", labelpad=12)
    plt.ylabel(r'$|\frac{d||w||_2}{dt}|$', labelpad=12)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

    plt.tight_layout()

    plt.xlim(left=0)

    plt.legend()

    train_fn = save_dir+'skip_weight_gradient_changes_doubled_SF.svg'
    plt.savefig(train_fn)
    # plt.show()

    # Convolutional weights

    # Convolutional weights
    
    all_c1_weights = []
    all_c2_weights = []
    all_c3_weights = []
    all_c4_weights = []
    all_c5_weights = []

    plt.figure(figsize=(5,5), dpi=300)

    for trial in trials:

        ref_ori = run[0]
        sf = run[1]
        sep = run[2]

        layer_sizes = [64*56*56,
                    64*27*27,
                    192*27*27,
                    192*13*13,
                    384*13*13,
                    256*13*13,
                    256*13*13,
                    256*6*6
                    ]

        conv1_file = np.load(weight_dir+'all_conv1_weights_ref_'+str(ref_ori)+'_sf_'+str(sf)+'_sep_'+str(sep)+'_lr_0.0001_trial_'+str(trial)+'.npy', allow_pickle=True)
        conv2_file = np.load(weight_dir+'all_conv2_weights_ref_'+str(ref_ori)+'_sf_'+str(sf)+'_sep_'+str(sep)+'_lr_0.0001_trial_'+str(trial)+'.npy', allow_pickle=True)
        conv3_file = np.load(weight_dir+'all_conv3_weights_ref_'+str(ref_ori)+'_sf_'+str(sf)+'_sep_'+str(sep)+'_lr_0.0001_trial_'+str(trial)+'.npy', allow_pickle=True)
        conv4_file = np.load(weight_dir+'all_conv4_weights_ref_'+str(ref_ori)+'_sf_'+str(sf)+'_sep_'+str(sep)+'_lr_0.0001_trial_'+str(trial)+'.npy', allow_pickle=True)
        conv5_file = np.load(weight_dir+'all_conv5_weights_ref_'+str(ref_ori)+'_sf_'+str(sf)+'_sep_'+str(sep)+'_lr_0.0001_trial_'+str(trial)+'.npy', allow_pickle=True)

        curr_index = 0
        c1_weights = []
        c2_weights = []
        c3_weights = []
        c4_weights = []
        c5_weights = []

        for c1, c2, c3, c4, c5 in zip(conv1_file, conv2_file, conv3_file, conv4_file, conv5_file):

            c1_weights.append(np.linalg.norm(c1))
            c2_weights.append(np.linalg.norm(c2))
            c3_weights.append(np.linalg.norm(c3))
            c4_weights.append(np.linalg.norm(c4))
            c5_weights.append(np.linalg.norm(c5))

        all_c1_weights.append(c1_weights)
        all_c2_weights.append(c2_weights)
        all_c3_weights.append(c3_weights)
        all_c4_weights.append(c4_weights)
        all_c5_weights.append(c5_weights)
            
    # Compute absolute gradients (rate of change) for convolutional weights
    conv_c1_rate = np.array([np.abs(np.gradient(np.array(c1_weights))) for c1_weights in all_c1_weights])
    conv_c2_rate = np.array([np.abs(np.gradient(np.array(c2_weights))) for c2_weights in all_c2_weights])
    conv_c3_rate = np.array([np.abs(np.gradient(np.array(c3_weights))) for c3_weights in all_c3_weights])
    conv_c4_rate = np.array([np.abs(np.gradient(np.array(c4_weights))) for c4_weights in all_c4_weights])
    conv_c5_rate = np.array([np.abs(np.gradient(np.array(c5_weights))) for c5_weights in all_c5_weights])
    
    conv_c1_mean = np.mean(conv_c1_rate, axis=0)
    conv_c2_mean = np.mean(conv_c2_rate, axis=0)
    conv_c3_mean = np.mean(conv_c3_rate, axis=0)
    conv_c4_mean = np.mean(conv_c4_rate, axis=0)
    conv_c5_mean = np.mean(conv_c5_rate, axis=0)
    
    conv_c1_std = sem(conv_c1_rate, axis=0)
    conv_c2_std = sem(conv_c2_rate, axis=0)
    conv_c3_std = sem(conv_c3_rate, axis=0)
    conv_c4_std = sem(conv_c4_rate, axis=0)
    conv_c5_std = sem(conv_c5_rate, axis=0)

    x_vals = np.arange(1,501,20)
    
    plt.plot(x_vals, conv_c1_mean, label='Conv1', lw=5, solid_capstyle='round', zorder=5)
    plt.plot(x_vals, conv_c2_mean, label='Conv2', lw=5, solid_capstyle='round', zorder=4)
    plt.plot(x_vals, conv_c3_mean, label='Conv3', lw=5, solid_capstyle='round', zorder=3)
    plt.plot(x_vals, conv_c4_mean, label='Conv4', lw=5, solid_capstyle='round', zorder=2)
    plt.plot(x_vals, conv_c5_mean, label='Conv5', lw=5, solid_capstyle='round', zorder=1)
    
    plt.fill_between(x_vals, conv_c1_mean-conv_c1_std, conv_c1_mean+conv_c1_std, alpha=0.3, zorder=5)
    plt.fill_between(x_vals, conv_c2_mean-conv_c2_std, conv_c2_mean+conv_c2_std, alpha=0.3, zorder=4)
    plt.fill_between(x_vals, conv_c3_mean-conv_c3_std, conv_c3_mean+conv_c3_std, alpha=0.3, zorder=3)
    plt.fill_between(x_vals, conv_c4_mean-conv_c4_std, conv_c4_mean+conv_c4_std, alpha=0.3, zorder=2)
    plt.fill_between(x_vals, conv_c5_mean-conv_c5_std, conv_c5_mean+conv_c5_std, alpha=0.3, zorder=1)

    plt.xlabel("Steps", labelpad=12)
    plt.ylabel(r'$|\frac{d||w||_2}{dt}|$', labelpad=12)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

    plt.tight_layout()

    plt.xlim(left=0)
    #plt.ylim(bottom=0)

    plt.legend()

    train_fn = save_dir+'conv_weight_gradient_changes_doubled_SF.svg'
    plt.savefig(train_fn)
    # plt.show()