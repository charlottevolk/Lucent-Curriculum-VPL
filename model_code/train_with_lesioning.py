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
from skip_alexnet import AlexNet
from statistics import mean
import os

if __name__ == '__main__':

    # set the device to use
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # set the parameters for the dataset
    test_root_dir = 'StimulusImages/SG_test_double_sf/sep_1.0/'
    ref_orientation = 0
    separation_angle = 10
    contrast = 1
    phase = 0
    spatial_freq = 10
    num_images = 100

    # set the parameters for the model
    num_classes = 2
    num_epochs = 5
    learning_rate = 0.0001

    batch_size = 1 # controls single-sample or multi-sample update
    num_images = 100
    num_transfer_images = 200

    if batch_size == 1:
        from dataset_single_example import GratingDataset
    else:
        from dataset_shuffled import GratingDataset

    # set the transforms
    stimulus_noise_sd = 0.02#25/255 # Caffe does noise in pixel values, torch does it in 0-1 tensor space
    confidence_noise_sd = 0.3#25/255
    confidence_noise_sd_NAME = 0.3

    data_transforms = transforms.Compose([
        transforms.Resize(227),
        transforms.ToTensor(),
        GaussianNoise(0, stimulus_noise_sd), # STANDARD DEVIATION OF GAUSSIAN NOISE
    ])
    
    num_neurons_range = list(range(0, 1001, 10))
    
    save_dir = 'saved_outputs/'
    trials = range(17,21)

    for trial_num in trials:

        for num_neurons in num_neurons_range:

            test_count = 0
            count = 0

            model = []
            seps = []

            def copy_weights(model, alexnet):
                model.conv1.weight.data = alexnet.features[0].weight.data
                model.conv1.bias.data = alexnet.features[0].bias.data
                model.conv2.weight.data = alexnet.features[3].weight.data
                model.conv2.bias.data = alexnet.features[3].bias.data
                model.conv3.weight.data = alexnet.features[6].weight.data
                model.conv3.bias.data = alexnet.features[6].bias.data
                model.conv4.weight.data = alexnet.features[8].weight.data
                model.conv4.bias.data = alexnet.features[8].bias.data
                model.conv5.weight.data = alexnet.features[10].weight.data
                model.conv5.bias.data = alexnet.features[10].bias.data

            def getActivation(name):
                def hook(model, input, output):
                    activation[name] = output.detach()
                return hook
            
            activation = {}


            test_losses = []
            train_losses = []
            train_accuracy = []
            test_accuracy = []

            # For loading old models
            model_dir = save_dir
            path = model_dir+'models/original_model_0_sf_0.05_sep_'+'1.0'+'_trial_'+str(trial_num)+'.pth'
            alexnet = AlexNet()
            alexnet.load_state_dict(torch.load(path))
            model.append(alexnet)
            model[count] = model[count].to(device)

            # Lesion neurons
            if num_neurons > 0:
                indices_to_lesion = np.loadtxt(save_dir+'data/max_abs_neurons_'+str(num_neurons)+'_sep_1.0_lr_0.0001_trial_'+str(trial_num)+'.csv', delimiter=',')[1]
                model[count].fc1.weight.data[0][indices_to_lesion] = 0.0

            test_sep = 1.0
            test_ref_ori = 0
            test_sf = 0.1

            test_ref_dir = 'StimulusImages/SG_refs/'
            test_ref_img = Image.open(test_ref_dir+'REFERENCE_ref_'+str(test_ref_ori)+'_sep_0.0_contr_1_ph_0.0_sf_'+str(0.1)+'_NONE.png') # ref image for testing set
            test_ref_img = data_transforms(test_ref_img).to(device)
            
            transfer_dataset = GratingDataset(test_root_dir, transform=data_transforms, ref_orientation=ref_orientation, separation_angle=separation_angle, contrast=contrast, phase=phase, spatial_freq=spatial_freq, num_images=num_transfer_images)
            transfer_dataloader = DataLoader(transfer_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

            dir_count = 0
            step_count = 0

            transfer_losses = []
            transfer_accuracies = []
            transfer_confidence = []
            transfer_noisy_confidence = []
                
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

                    model[count].eval()

                    test_inputs, test_labels = test_data
                    test_inputs = test_inputs.to(device)
                    test_labels = test_labels.to(device)
                    
                    test_outputs = model[count](test_inputs, test_ref_img, test_sep)
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

                    np.savetxt(save_dir+'data/'+str(num_neurons)+'_neurons_transfer_data_ref_'+str(test_ref_ori)+'_sf_'+str(test_sf)+'_sep_'+str(test_sep)+'_lr_0.0001_trial_'+str(trial_num)+'.csv', transfer_losses, delimiter=",")
                    np.savetxt(save_dir+'data/'+str(num_neurons)+'_neurons_transfer_accuracy_ref_'+str(test_ref_ori)+'_sf_'+str(test_sf)+'_sep_'+str(test_sep)+'_lr_0.0001_trial_'+str(trial_num)+'.csv', transfer_accuracies, delimiter=",")
                    np.savetxt(save_dir+'data/'+str(num_neurons)+'_neurons_transfer_confidence_ref_'+str(test_ref_ori)+'_sf_'+str(test_sf)+'_sep_'+str(test_sep)+'_lr_0.0001_trial_'+str(trial_num)+'.csv', transfer_confidence, delimiter=",")
                    np.savetxt(save_dir+'data/'+str(num_neurons)+'_neurons_transfer_noisy_confidence_ref_'+str(test_ref_ori)+'_sf_'+str(test_sf)+'_sep_'+str(test_sep)+'_lr_0.0001_trial_'+str(trial_num)+'.csv', transfer_noisy_confidence, delimiter=",")
        print('Finished trial '+str(trial_num)+' '+str(num_neurons)+' neurons')
