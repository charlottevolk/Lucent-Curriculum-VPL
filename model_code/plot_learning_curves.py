import matplotlib.pyplot as plt
import csv
import numpy as np
import matplotlib as mpl
from cycler import cycler
from matplotlib.cm import get_cmap
import seaborn as sns
import os
from scipy.stats import sem

SMALLEST_SIZE = 18
SMALL_SIZE = 20
TITLE_SIZE = 20
MEDIUM_SIZE = 24
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALLEST_SIZE)    # legend fontsize
#plt.rc('title', fontsize=MEDIUM_SIZE)  # fontsize of the figure title

sns.set_palette("GnBu_r")

model_names = ['AlexNet_skip']
for model_name in model_names:

    base_dir = 'saved_outputs/<YOUR_FOLDER_NAME>/'  

    deg = u'\N{DEGREE SIGN}'

    plot_SEM = False
    transfer_condition = 'SF' # or SF

    def read_data(dir, filename):
        file = open(dir+filename)
        reader = csv.reader(file)
        data = []
        for row in reader: data.append(row[0])
        data = np.array(data)
        data = data.astype(np.float64)
        return data

    iterations = np.arange(1,501)#, 50)

    # Training learning curve plot

    if transfer_condition == 'ref_ori':
        spatial_freq_train = 0.05
        spatial_freq_test = 0.05
        ref_angle_train = 0
        ref_angle_test = 15
    elif transfer_condition == 'SF':
        spatial_freq_train = 0.05
        spatial_freq_test = 0.1
        ref_angle_train = 0
        ref_angle_test = 0

    #[ref ori, sf, sep]
    angle_seps = [0.5, 1.0, 2.0, 5.0, 10.0]

    save_dirs = [base_dir+f'{angle_seps[i]}_angle_sep/' for i in range(len(angle_seps))]

    trials = range(1,101)

    to_plot = ['train', 'transfer']

    for t in to_plot:
        order_count = 5  # Reset for each plot
        
        if t == 'train':
            plt.figure(figsize=(5,5), dpi=300)
            plt.xlabel("Steps", labelpad = 12)
            plt.ylabel("Train accuracy", labelpad = 12)
        if t == 'transfer':
            plt.figure(figsize=(5,5), dpi=300)
            plt.xlabel("Angle separations", labelpad = 12)
            plt.ylabel("Transfer accuracy", labelpad = 12)

        for sep, save_dir in zip(angle_seps, save_dirs):
            c = sns.color_palette()[5-order_count]

            train_data = []
            transfer_data = []
            for trial in trials:
                train_data_1 = read_data(save_dir, 'data/train_accuracy_ref_'+str(ref_angle_train)+'_sf_'+str(spatial_freq_train)+'_sep_'+str(sep)+'_lr_0.0001_trial_'+str(trial)+'.csv')
                transfer_data_1 = read_data(save_dir,'data/transfer_accuracy_ref_'+str(ref_angle_test)+'_sf_'+str(spatial_freq_test)+'_sep_'+str(sep)+'_lr_0.0001_trial_'+str(trial)+'.csv')
                transfer_mean = np.mean(transfer_data_1)
                train_data.append(train_data_1)
                transfer_data.append(transfer_mean)
            
            train_mean = np.array(train_data).mean(axis=0)#[iterations]
            train_sem = sem(train_data, axis=0)#[iterations]
            transfer_mean = np.array(transfer_data).mean(axis=0)
            transfer_sem = sem(transfer_data, axis=0)

            if t == 'train':
                plt.plot(iterations, train_mean, label=str(sep)+deg, lw=5, solid_capstyle='round', zorder=order_count, color=c)
                if plot_SEM:
                    plt.fill_between(iterations, train_mean-train_sem, train_mean+train_sem, alpha=0.2, color=c, interpolate=True)
            if t == 'transfer':
                plt.scatter(str(sep), transfer_mean, s=100, zorder=order_count, color=c, label=str(sep)+deg)
                plt.errorbar(str(sep), transfer_mean, yerr=transfer_sem, capsize=3, color=c)
            order_count -= 1

        save_dir = base_dir+'plots/'
        if not os.path.exists(save_dir): os.makedirs(save_dir)

        if plot_SEM and t == 'train':
            train_fn = save_dir+'train_accuracy_learning_curve_with_SEM.svg'
        elif t == 'train':
            train_fn = save_dir+'train_accuracy_learning_curve.svg'
        if t == 'transfer':
            train_fn = save_dir+'transfer_accuracy_with_SEM.svg'

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.legend()

        plt.savefig(train_fn)
        # plt.show()