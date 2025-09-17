import matplotlib.pyplot as plt
import csv
import numpy as np
import matplotlib as mpl
from cycler import cycler
from matplotlib.cm import get_cmap
import seaborn as sns

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

save_dir = 'saved_outputs/<YOUR_FOLDER_NAME>/'      

deg = u'\N{DEGREE SIGN}'

plot_SEM = False

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

#[ref ori, sf, sep]
train_params = [
              [0,0.05,0.5],
              [0,0.05,1.0],
              [0,0.05,2.0],
              [0,0.05,5.0],
              [0,0.05,10.0]
              ]

order_count = 5

trials = range(1,101)

to_plot = ['train', 'transfer']

for t in to_plot:
    if t == 'train':
        plt.figure(figsize=(5,5), dpi=300)
        plt.xlabel("Steps", labelpad = 12)
        plt.ylabel("Train accuracy", labelpad = 12)
    if t == 'transfer':
        plt.figure(figsize=(5,5), dpi=300)
        plt.xlabel("Angle separations", labelpad = 12)
        plt.ylabel("Transfer accuracy", labelpad = 12)

    for run in train_params:
        ref_ori = run[0]
        sf = run[1]
        sep = run[2]
        c = sns.color_palette()[5-order_count]

        train_data = []
        transfer_data = []
        for trial in trials:
            train_data_1 = read_data(save_dir, 'data/skip/SF_doubled_lr_0.0001/train_accuracy_ref_'+str(ref_ori)+'_sf_'+str(sf)+'_sep_'+str(sep)+'_lr_0.0001_trial_'+str(trial)+'.csv')
            transfer_data_1 = read_data(save_dir,'data/skip/SF_doubled_lr_0.0001/transfer_accuracy_ref_'+str(ref_ori)+'_sf_'+str(0.1)+'_sep_'+str(sep)+'_lr_0.0001_trial_'+str(trial)+'.csv')
            transfer_mean = np.mean(transfer_data_1)
            train_data.append(train_data_1)
            transfer_data.append(transfer_mean)
        
        train_mean = np.array(train_data).mean(axis=0)#[iterations]
        train_std = np.array(train_data).std(axis=0)#[iterations]
        transfer_mean = np.array(transfer_data).mean(axis=0)
        transfer_std = np.array(transfer_data).std(axis=0) 

        if t == 'train':
            plt.plot(iterations, train_mean, label=str(sep)+deg, lw=5, solid_capstyle='round', zorder=order_count, color=c)
            if plot_SEM:
                plt.fill_between(iterations, train_mean-train_std, train_mean+train_std, alpha=0.2, color=c, interpolate=True)
        if t == 'transfer':
            plt.scatter(str(sep), transfer_mean, s=100, zorder=order_count, color=c, label=str(sep)+deg)
            plt.errorbar(str(sep), transfer_mean, yerr=transfer_std, capsize=3, color=c)
        order_count -= 1

    save_dir = 'saved_outputs/plots/'
    if t == 'train':
        train_fn = save_dir+'train_accuracy_learning_curve.svg'
    if t == 'transfer':
        train_fn = save_dir+'transfer_accuracy.svg'

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.legend()

    plt.savefig(train_fn)
    # plt.show()