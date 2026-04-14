import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv
from rastermap import Rastermap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from umap import UMAP
from models.skip_alexnet import AlexNet
import torch
from scipy.stats import zscore

# Plot settings
SMALLEST_SIZE = 18
SMALL_SIZE = 22
TITLE_SIZE = 20
MEDIUM_SIZE = 26
BIGGER_SIZE = 12
plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=SMALL_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALLEST_SIZE)
# sns.set_palette("colorblind")

def read_data(dir, filename):
    with open(dir + filename) as file:
        reader = csv.reader(file)
        data = [row[0] for row in reader]
    return np.array(data, dtype=np.float64)

# Define paths
base_dir = 'saved_outputs/'
nonseq_dir = base_dir + 'nonsequential_doubled_SF_AlexNet/'
seq_dir = base_dir + 'sequential_doubled_SF_AlexNet/'
shuffled_dir = base_dir + 'shuffled_doubled_SF_AlexNet/'

activation_dir = base_dir + 'collecting_activations_imagenet_100_AlexNet/'
save_dir = base_dir + 'plots/'
if not os.path.exists(save_dir): os.makedirs(save_dir)

models = ['Non-seq', 'Shuff', 'Seq']

colors = sns.color_palette("colorblind", 3)
custom_palette = {
    "Non-seq": colors[0],
    "Shuff": colors[1],
    "Seq": colors[2]
}

trials = range(1, 2)

stimulus_noise_sd = 0.02
confidence_noise_sd = 0.3

dirs = [nonseq_dir, shuffled_dir, seq_dir]

epsilon = 1e-8

sep = 1.0

all_model_activations = []
all_model_labels = []

for model_dir, model in zip(dirs, models):
    print(f"Processing model: {model}")

    all_activations = []

    for trial in trials:
        print(f"  Trial {trial}")
        activations_all = np.load(activation_dir + f'activations/100_imgs_all_activations_no_noise_1000_imagenet_0_sf_0.05_sep_0.5_lr_0.0001_model_trial_{trial}_batch_size_1.npy')
        activations_all = np.float64(activations_all)
        
        path = model_dir + f'models/original_model_0_sf_0.05_sep_{sep}_trial_{trial}.pth'
        alexnet = AlexNet()
        alexnet.load_state_dict(torch.load(path, map_location='cpu'))
        readout_weights = alexnet.fc1.weight.data[0]

        num_neurons = 150
        source_path = model_dir + f'data/max_abs_neurons_{num_neurons}_sep_{sep}_lr_0.0001_trial_{trial}.csv'
        neuron_indices = np.loadtxt(source_path, delimiter=",", dtype=int)[1]
        sorted_indices = np.argsort(neuron_indices)
        important_neurons = neuron_indices[sorted_indices]

        # Extract and normalize activations
        imp_activations = activations_all[:, important_neurons]
        all_activations.append(imp_activations)

    all_activations = np.concatenate((all_activations), axis=0)
    all_activations = all_activations.T
    all_model_activations.append(all_activations)
    all_model_labels.append(np.full(all_activations.shape[0], model))
   
all_model_labels = np.concatenate(all_model_labels, axis=0)
all_model_activations = np.concatenate([
    StandardScaler().fit_transform(act) for act in all_model_activations
])

# PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(all_model_activations)

# Plot PCA - all models together
plt.figure(figsize=(12, 5), dpi=150)
sns.scatterplot(
    x=pca_result[:, 0],
    y=pca_result[:, 1],
    hue=all_model_labels,
    palette=custom_palette,
    hue_order=models,
    # alpha=0.7
)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend()
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f"{save_dir}/pca_activations.svg", bbox_inches='tight')