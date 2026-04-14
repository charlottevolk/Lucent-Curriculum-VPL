# train the alexnet model (from test_alexnet.py) on the grating dataset
# load the model and the dataset
# OPTIMIZED VERSION: caches test data, uses vectorized operations, removes redundant shuffling

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils
from torchvision import datasets, models, transforms
import torchvision.transforms.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
import random
import math
#import cv2
import PIL
from PIL import Image
import scipy
from scipy import ndimage
from transforms import GaussianNoise
from models.skip_alexnet import AlexNet
from statistics import mean
import os

if __name__ == '__main__':

    base_dir = 'saved_outputs/'
    seq_dir = base_dir + 'sequential_doubled_SF_AlexNet/'
    shuff_dir = base_dir + 'shuffled_doubled_SF_AlexNet/'
    nonseq_dir = base_dir + 'nonsequential_doubled_SF_AlexNet/'
    forced_seq_dirs = [base_dir + f'forced_sequential_doubled_SF_AlexNet_factor_{factor}/' for factor in factors]
    forced_antiseq_dirs = [base_dir + f'forced_antisequential_doubled_SF_AlexNet_factor_{factor}/' for factor in factors]

    folders_to_lesion_from = [seq_dir, shuff_dir, nonseq_dir] + forced_seq_dirs + forced_antiseq_dirs

    # set the device to use
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # set the parameters for the dataset
    test_root_dir = 'StimulusImages/SG_test_double_sf/sep_1.0/'
    num_images = 100

    # set the parameters for the model
    num_classes = 2
    num_epochs = 5
    learning_rate = 0.0001

    batch_size = 1 # controls single-sample or multi-sample update
    num_images = 100
    num_transfer_images = 200  # Use full test set for more reliable neuron importance measurements

    if batch_size == 1:
        from dataset_single_example import GratingDataset
    else:
        from dataset_shuffled import GratingDataset

    # set the transforms
    stimulus_noise_sd = 0.02#25/255 # Caffe does noise in pixel values, torch does it in 0-1 tensor space
    confidence_noise_sd = 0.3#25/255

    data_transforms = transforms.Compose([
        transforms.Resize(227),
        transforms.ToTensor(),
        GaussianNoise(0, stimulus_noise_sd), # STANDARD DEVIATION OF GAUSSIAN NOISE
    ])
    
    num_neurons = 1000
    epsilon = 1e-3
    
    trials = range(1,21)

    # OPTIMIZATION 1: Cache test data - load once before the loops
    test_sep = 1.0
    test_ref_ori = 0
    test_sf = 0.1

    test_ref_dir = 'StimulusImages/SG_refs/'
    test_ref_img = Image.open(test_ref_dir+'REFERENCE_ref_'+str(test_ref_ori)+'_sep_0.0_contr_1_ph_0.0_sf_'+str(test_sf)+'_NONE.png')
    test_ref_img = data_transforms(test_ref_img).to(device)
    
    transfer_dataset = GratingDataset(test_root_dir, transform=data_transforms, num_images=num_transfer_images)
    # OPTIMIZATION 3: Remove redundant shuffling - set shuffle=False for evaluation data
    transfer_dataloader = DataLoader(transfer_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Cache all test data in memory
    cached_test_data = []
    for test_data in transfer_dataloader:
        if batch_size == 1:
            test_data = list([test_data[0], test_data[1]])
        else:
            test_data = list([torch.cat(torch.unbind(test_data[0], 1)), torch.cat(torch.unbind(test_data[1], 1))])
        
        test_inputs, test_labels = test_data
        test_inputs = test_inputs.to(device)
        test_labels = test_labels.to(device).unsqueeze(1).float()
        cached_test_data.append((test_inputs, test_labels))
    
    print(f"Cached {len(cached_test_data)} batches of test data")

    criterion = nn.BCELoss()

    for folder in folders_to_lesion_from:
        print('Lesioning from folder: '+folder)
        save_dir = base_dir + folder

        for trial_num in trials:

            test_count = 0
            count = 0

            model = []
            seps = []

            test_losses = []
            train_losses = []
            train_accuracy = []
            test_accuracy = []

            # For loading old models
            model_dir = base_dir + folder
            path = model_dir+'models/original_model_0_sf_0.05_sep_'+'1.0'+'_trial_'+str(trial_num)+'.pth'
            alexnet = AlexNet()
            alexnet.load_state_dict(torch.load(path))
            model.append(alexnet)
            model[count] = model[count].to(device)
            
            neuron_count = 0
            neurons = {}
            
            # OPTIMIZATION 2: Get weight information once using vectorized operations
            fc1_weights = model[count].fc1.weight.data[0].clone()
            abs_weights = torch.abs(fc1_weights)
            
            # Lesion neurons with highest absolute contributions
            print(model[count].fc1.weight.data[0].shape[0])
            for neuron_index in range(0, model[count].fc1.weight.data[0].shape[0]):
                abs_value = abs_weights[neuron_index].item()
                value_with_sign = fc1_weights[neuron_index].item()

                # We don't care about the neuron's contribution if it's zero or very small, skip to next neuron
                if abs_value < epsilon: continue
                else: model[count].fc1.weight.data[0][neuron_index] = 0.0

                dir_count = 0
                step_count = 0

                transfer_losses = []
                transfer_accuracies = []
                transfer_confidence = []
                transfer_noisy_confidence = []
                    
                with torch.no_grad():
                    # OPTIMIZATION 1: Use cached test data instead of reloading
                    for test_inputs, test_labels in cached_test_data:
                        ############## TESTING ##############

                        model[count].eval()
                        
                        test_outputs = model[count](test_inputs, test_ref_img, test_sep)
                        test_loss = criterion(test_outputs[0], test_labels)
                        transfer_losses.append(test_loss.item())

                        # OPTIMIZATION 2: Use vectorized operations for confidence calculations
                        confidence = test_outputs[0]
                        noisy_confidence = torch.add(confidence, torch.randn_like(confidence) * confidence_noise_sd)
                        # Clamp output to 0-1 (probability range)
                        noisy_confidence = torch.clamp(noisy_confidence, 0, 1)
                        transfer_confidence.append(torch.mean(torch.abs(confidence-0.5)*2).cpu())
                        transfer_noisy_confidence.append(torch.mean(torch.abs(noisy_confidence-0.5)*2).cpu())

                        # OPTIMIZATION 2: Vectorized accuracy calculation
                        output_labels = (noisy_confidence >= 0.5).float()
                        test_acc = (output_labels == test_labels).float().mean().item()
                        transfer_accuracies.append(test_acc)

                        test_summed_acc = mean(transfer_accuracies)

                test_loss_avg = mean(transfer_losses)
                test_acc_avg = mean(transfer_accuracies)
                neurons[neuron_index] = (test_loss_avg, test_acc_avg) # Add performance to dictionary, indexed by neuron lesioned

                neuron_count += 1

                # Reset neuron weight back to original value before lesioning next neuron
                model[count].fc1.weight.data[0][neuron_index] = value_with_sign
            print('Finished trial '+str(trial_num)+' '+str(num_neurons)+' neurons')

            # Get 1000 neurons with most effect on performance when lesioned i.e. lowest test accuracy
            # Sort by accuracy first, then by loss to break ties
            sorted_neurons = sorted(neurons.items(), key=lambda x: (x[1][1], x[1][0])) # sort by (accuracy, loss)
            top_neurons = sorted_neurons[:num_neurons] # get lowest accuracy ones

            # Convert to 2D arrays for saving: [neuron_index, test_loss, test_acc]
            top_neurons_array = np.array([[neuron_idx, loss, acc] for neuron_idx, (loss, acc) in top_neurons])
            sorted_neurons_array = np.array([[neuron_idx, loss, acc] for neuron_idx, (loss, acc) in sorted_neurons])

            # Save the top neurons to lesion - save the full dictionary with the neuron index, loss and accuracy
            np.savetxt(save_dir+'data/'+f'lesioning_by_contribution_eps_{epsilon}_{num_transfer_images}_imgs_'+str(num_neurons)+'_neurons_dictionary_ref_'+str(test_ref_ori)+'_sf_'+str(test_sf)+'_sep_'+str(test_sep)+'_lr_0.0001_trial_'+str(trial_num)+'.csv', top_neurons_array, delimiter=",")

            # Save all other neurons' performance as well for analysis, sorted by accuracy
            np.savetxt(save_dir+'data/'+f'lesioning_by_contribution_eps_{epsilon}_{num_transfer_images}_imgs_all_neurons_dictionary_ref_'+str(test_ref_ori)+'_sf_'+str(test_sf)+'_sep_'+str(test_sep)+'_lr_0.0001_trial_'+str(trial_num)+'.csv', sorted_neurons_array, delimiter=",")
