# train the alexnet model (from test_alexnet.py) on the grating dataset
# load the model and the dataset

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
from copy_weights import copy_weights_alexnet

if __name__ == '__main__':

    base_dir = 'saved_outputs/'
    seq_dir = base_dir + 'sequential_doubled_SF_AlexNet/'
    shuff_dir = base_dir + 'shuffled_doubled_SF_AlexNet/'
    nonseq_dir = base_dir + 'nonsequential_doubled_SF_AlexNet/'

    transfer_condition = 'SF' # or 'ref_ori'
    
    folders_to_lesion_from = [seq_dir, shuff_dir, nonseq_dir]

    # set the device to use
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # set the parameters for the model
    num_classes = 2
    num_epochs = 5
    learning_rate = 0.0001

    batch_size = 1 # controls single-sample or multi-sample update
    num_transfer_images = 200

    if transfer_condition == 'ref_ori':
        root_dir = 'StimulusImages/SG_train_double_sf/'
        test_root_dir = 'StimulusImages/SG_test_changed_ref_15/sep_1.0/'
        spatial_freq_train = 0.05
        spatial_freq_test = 0.05
        ref_angle_train = 0
        ref_angle_test = 15
    elif transfer_condition == 'SF':
        root_dir = 'StimulusImages/SG_train_double_sf/'
        test_root_dir = 'StimulusImages/SG_test_double_sf/sep_1.0/'
        spatial_freq_train = 0.05
        spatial_freq_test = 0.1
        ref_angle_train = 0
        ref_angle_test = 0

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
    
    num_neurons_range = list(range(0, 251, 10))

    trials = range(1,21)

    for model_name, folder in zip(model_names, folders_to_lesion_from):
        print('Lesioning from folder: '+folder)
        sep_names = ['0.5', '1.0', '2.0', '5.0', '10.0']
        # sep_names = ['10.0']
        save_dirs = [base_dir + folder + f'{sep}_angle_sep/' for sep in sep_names]
        for sep, save_dir in zip(sep_names, save_dirs):
            print('Testing lesioning with sep: '+sep)

            for trial_num in trials:

                for num_neurons in num_neurons_range:

                    test_count = 0
                    count = 0

                    model = []
                    seps = []

                    test_losses = []
                    train_losses = []
                    train_accuracy = []
                    test_accuracy = []

                    test_sep = sep

                    path = save_dir+'models/original_model_0_sf_0.05_sep_'+str(sep)+'_trial_'+str(trial_num)+'.pth'
                    model = AlexNet()
                    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
                    model = model.to(device)

                    # Lesion neurons
                    if num_neurons > 0:
                        indices_to_lesion = np.loadtxt(save_dir+'data/max_abs_neurons_'+str(num_neurons)+'_sep_'+str(test_sep)+'_lr_0.0001_trial_'+str(trial_num)+'.csv', delimiter=',')[1]
                        model.fc1.weight.data[0][indices_to_lesion] = 0.0

                    test_ref_dir = 'StimulusImages/SG_refs/'
                    test_ref_img = Image.open(test_ref_dir+'REFERENCE_ref_'+str(ref_angle_test)+'_sep_0.0_contr_1_ph_0.0_sf_'+str(spatial_freq_test)+'_NONE.png') # ref image for testing set
                    test_ref_img = data_transforms(test_ref_img).to(device)
                    
                    transfer_dataset = GratingDataset(test_root_dir, transform=data_transforms, num_images=num_transfer_images)
                    transfer_dataloader = DataLoader(transfer_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

                    dir_count = 0
                    step_count = 0

                    transfer_losses = []
                    transfer_accuracies = []
                    transfer_confidence = []
                    transfer_noisy_confidence = []

                    criterion = nn.BCELoss()
                        
                    with torch.no_grad():
                        for (test_i, test_data) in enumerate(transfer_dataloader):
                            if batch_size == 1:
                                # print(train_data[0].shape)
                                test_data = list([test_data[0], test_data[1]])
                            else:
                                test_data = list([torch.cat(torch.unbind(test_data[0], 1)), torch.cat(torch.unbind(test_data[1], 1))])
                            
                            test_idx = torch.randperm(test_data[0].shape[0])
                            test_data[0] = test_data[0][test_idx].view(test_data[0].size())
                            test_data[1] = test_data[1][test_idx].view(test_data[1].size())

                            ############## TESTING ##############

                            model.eval()

                            test_inputs, test_labels = test_data
                            test_inputs = test_inputs.to(device)
                            test_labels = test_labels.to(device)
                            
                            test_outputs = model(test_inputs, test_ref_img, test_sep)
                            test_labels = test_labels.unsqueeze(1)
                            test_labels = test_labels.float()
                            test_loss = criterion(test_outputs[0], test_labels)
                            transfer_losses.append(test_loss.item())

                            # Add noise to confidence before calculating accuracy
                            confidence = test_outputs[0]
                            noisy_confidence = torch.add(confidence, torch.randn_like(confidence) * confidence_noise_sd)
                            # Clamp output to 0-1 (probability range)
                            noisy_confidence = torch.clamp(noisy_confidence, 0, 1)
                            transfer_confidence.append(torch.mean(torch.abs(confidence-0.5)*2).cpu())
                            transfer_noisy_confidence.append(torch.mean(torch.abs(noisy_confidence-0.5)*2).cpu())

                            # Calculate testing accuracy for this iteration
                            output_labels = []
                            for item in noisy_confidence:
                                if item >= 0.5: output_labels.append(float(1))
                                else: output_labels.append(float(0))
                            test_acc = [output_labels[i] == test_labels[i] for i in range(0,len(output_labels))]
                            test_acc = [float(x) for x in test_acc]
                            test_acc = mean(test_acc)
                            transfer_accuracies.append(test_acc)

                            test_summed_acc = mean(transfer_accuracies)

                    np.savetxt(save_dir+'data/'+'lesioning_'+str(num_neurons)+'_neurons_transfer_data_ref_'+str(ref_angle_test)+'_sf_'+str(spatial_freq_test)+'_sep_'+str(test_sep)+'_lr_0.0001_trial_'+str(trial_num)+'.csv', transfer_losses, delimiter=",")
                    np.savetxt(save_dir+'data/'+'lesioning_'+str(num_neurons)+'_neurons_transfer_accuracy_ref_'+str(ref_angle_test)+'_sf_'+str(spatial_freq_test)+'_sep_'+str(test_sep)+'_lr_0.0001_trial_'+str(trial_num)+'.csv', transfer_accuracies, delimiter=",")
                    np.savetxt(save_dir+'data/'+'lesioning_'+str(num_neurons)+'_neurons_transfer_confidence_ref_'+str(ref_angle_test)+'_sf_'+str(spatial_freq_test)+'_sep_'+str(test_sep)+'_lr_0.0001_trial_'+str(trial_num)+'.csv', transfer_confidence, delimiter=",")
                    np.savetxt(save_dir+'data/'+'lesioning_'+str(num_neurons)+'_neurons_transfer_noisy_confidence_ref_'+str(ref_angle_test)+'_sf_'+str(spatial_freq_test)+'_sep_'+str(test_sep)+'_lr_0.0001_trial_'+str(trial_num)+'.csv', transfer_noisy_confidence, delimiter=",")
                print('Finished trial '+str(trial_num)+' '+str(num_neurons)+' neurons')
