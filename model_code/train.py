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
import shutil
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
    root_dir = 'StimulusImages/SG_train_double_sf/'
    test_root_dir = 'StimulusImages/SG_test_double_sf/'
    ref_orientation = 0
    separation_angle = 10
    contrast = 1
    phase = 0
    spatial_freq = 10
    num_images = 100
    num_transfer_images = 200

    # set the parameters for the model
    num_classes = 2
    num_epochs = 5
    learning_rate = 0.0001

    batch_size = 1 # controls single-sample or multi-sample update
    sequential = True

    pretrained = False
    collecting_gradients = False
    cyclicLR = False
    forced_sequential = False
    forced_antisequential = False
    factor = 6.0 # controls factor to modulate gradients for forced sequential update
    num_steps = 25
    frozen_readout = False
    frozen_conv = False

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

    # format: [[train/test]root_dir + [dir with correct sep angle], ref orientation, spatial freq, sep_angle]
    dir_list = [
                #[root_dir+'sep_0.5',0,0.05,0.5],
                #[root_dir+'sep_0.5',0,0.05,0.5],
                #[root_dir+'sep_0.5',0,0.05,0.5]
                #[root_dir+'sep_0.5',0,0.05,0.5],
                #[root_dir+'sep_0.5',0,0.05,0.5],
                #[root_dir+'sep_1.0',0,0.05,1.0],
                #[root_dir+'sep_2.0',0,0.05,2.0]
                # [root_dir+'sep_5.0',0,0.05,5.0],
                # [root_dir+'sep_10.0',0,0.05,10.0]
                [root_dir+'sep_10.0',0,0.05,10.0],
                [root_dir+'sep_5.0',0,0.05,5.0],
                [root_dir+'sep_2.0',0,0.05,2.0],
                [root_dir+'sep_1.0',0,0.05,1.0],
                [root_dir+'sep_0.5',0,0.05,0.5],
                # [root_dir+'sep_5.0_1.0_combined',0,0.05,5.0],
                # [root_dir+'sep_5.0_1.0_combined',0,0.05,1.0],
                # [root_dir+'sep_5.0',0,0.05,5.0],
                # [root_dir+'sep_1.0',0,0.05,1.0],
                # [root_dir+'sep_1.0',0,0.05,5.0],
                # [root_dir+'sep_1.0',0,0.05,1.0],
                # [root_dir+'sep_45.0',0,0.05,45.0]
                ]
    
    test_dir_list = [
                    #[test_root_dir+'sep_0.5',0,0.1,0.5],
                    #[test_root_dir+'sep_0.5',0,0.1,0.5],
                    #[test_root_dir+'sep_0.5',0,0.1,0.5]
                    #[test_root_dir+'sep_0.5',0,0.1,0.5],
                    #[test_root_dir+'sep_0.5',0,0.1,0.5],
                    #[test_root_dir+'sep_1.0',0,0.1,1.0],
                    #[test_root_dir+'sep_2.0',0,0.1,2.0]
                    # [test_root_dir+'sep_5.0',0,0.1,5.0],
                    # [test_root_dir+'sep_10.0',0,0.1,10.0]
                    [test_root_dir+'sep_10.0',0,0.1,10.0],
                    [test_root_dir+'sep_5.0',0,0.1,5.0],
                    [test_root_dir+'sep_2.0',0,0.1,2.0],
                    [test_root_dir+'sep_1.0',0,0.1,1.0],
                    [test_root_dir+'sep_0.5',0,0.1,0.5],
                    # [test_root_dir+'sep_1.0',0,0.1,5.0],
                    # [test_root_dir+'sep_1.0',0,0.1,1.0],
                    # [test_root_dir+'sep_45.0',0,0.1,45.0]
                    ]
    
    # num_pretraining_steps = 20
    if forced_sequential: forced_name = 'sequential'
    if forced_antisequential: forced_name = 'antisequential'

    # Replace with the directory where you want to save the outputs
    save_dir = 'saved_outputs/'
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    if not os.path.exists(save_dir+'models/'): os.makedirs(save_dir+'models/')
    if not os.path.exists(save_dir+'data/'): os.makedirs(save_dir+'data/')
    if not os.path.exists(save_dir+'weights/'): os.makedirs(save_dir+'weights/')
    
    # Number of models to train
    trials = range(1,21)

    # Where to load readout weight subspace to train model with frozen readout from
    if frozen_readout:
        forced_name = 'sequential'
        frozen_source_model_dir = 'saved_outputs/models/best_model/' # set here

    for trial_num in trials:

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

        if sequential:
            if not pretrained:
                # For new models
                model.append(AlexNet())
                alexnet = torchvision.models.alexnet(pretrained=True)
                copy_weights(model[count], alexnet)
                model[count] = model[count].to(device)
            
            else:
                # For loading old models - If you want to use a pretrained model
                model_dir = save_dir
                path = model_dir+'models/SF_doubled_lr_0.0001/original_model_0_sf_0.05_sep_'+'45.0'+'_trial_'+str(trial_num)+'.pth'
                alexnet = AlexNet()
                alexnet.load_state_dict(torch.load(path))
                model.append(alexnet)
                model[count] = model[count].to(device)

        # Use the readout subspace from another model to train the model
        if frozen_readout:

            # Load high weight neurons from sequential to transplant
            # data stored as: np.row_stack((max_values, indices, values_with_signs))
            num_neurons = 150
            
            source_path = frozen_source_model_dir+'data/max_abs_neurons_'+str(num_neurons)+'_sep_1.0_lr_0.0001_trial_'+str(trial_num)+'.csv'
            new_indices = np.loadtxt(source_path, delimiter=",", dtype=float)[1]
            new_values = np.loadtxt(source_path, delimiter=",", dtype=float)[2]

            other_indices = np.arange(0,580416)
            other_indices = np.ndarray.tolist(other_indices)
            other_indices = np.setdiff1d(other_indices, new_indices)

            # Save weights to freeze - freeze everything except for the transplanted readout subspace
            fc_weights = torch.index_select(model[count].fc1.weight.data, dim=1, index=torch.IntTensor(other_indices).to(device))


        # MAIN TRAINING LOOP    
        dir_count = 0
        step_count = 0
        for dir_num, (train_dir, test_dir) in enumerate(zip(dir_list, test_dir_list)):
            train_losses.append([])
            test_losses.append([])
            train_accuracy = []
            test_accuracy = []
            train_confidence = []
            test_confidence = []
            train_noisy_confidence = []
            test_noisy_confidence = []

            if not sequential:
                #For new models - non sequential
                model.append(AlexNet())
                alexnet = torchvision.models.alexnet(pretrained=True)
                copy_weights(model[count], alexnet)
                model[count] = model[count].to(device)

            if frozen_conv:
                for param in model[count].parameters():
                    param.requires_grad = False
                
                model[count].fc1.weight.requires_grad = True
                model[count].fc1.bias.requires_grad = True

            # store initial weights for each model

            # # Regular weights:
            init_conv1_weights = model[count].conv1.weight.data
            init_conv1_weights = init_conv1_weights.detach().clone()

            init_conv2_weights = model[count].conv2.weight.data
            init_conv2_weights = init_conv2_weights.detach().clone()

            init_conv3_weights = model[count].conv3.weight.data
            init_conv3_weights = init_conv3_weights.detach().clone()

            init_conv4_weights = model[count].conv4.weight.data
            init_conv4_weights = init_conv4_weights.detach().clone()

            init_conv5_weights = model[count].conv5.weight.data
            init_conv5_weights = init_conv5_weights.detach().clone()

            # # Skip connection weights:
            init_fc_weights = model[count].fc1.weight.data
            init_fc_weights = init_fc_weights.detach().clone()

            # Save initial weights

            train_ref_ori = train_dir[1]
            train_sf = train_dir[2]
            train_sep = train_dir[3]

            #Save skip weights
            np.savetxt(save_dir+'weights/init_fc_weights_ref_'+str(train_ref_ori)+'_sf_'+str(train_sf)+'_sep_'+str(train_sep)+'_lr_0.0001_trial_'+str(trial_num)+'.csv', init_fc_weights.cpu(), delimiter=",")

            # Save regular weights
            np.save(save_dir+'weights/conv1/init_conv1_weights_ref_'+str(train_ref_ori)+'_sf_'+str(train_sf)+'_sep_'+str(train_sep)+'_lr_0.0001_trial_'+str(trial_num), init_conv1_weights.cpu())

            np.save(save_dir+'weights/conv2/init_conv2_weights_ref_'+str(train_ref_ori)+'_sf_'+str(train_sf)+'_sep_'+str(train_sep)+'_lr_0.0001_trial_'+str(trial_num), init_conv2_weights.cpu())

            np.save(save_dir+'weights/conv3/init_conv3_weights_ref_'+str(train_ref_ori)+'_sf_'+str(train_sf)+'_sep_'+str(train_sep)+'_lr_0.0001_trial_'+str(trial_num), init_conv3_weights.cpu())

            np.save(save_dir+'weights/conv4/init_conv4_weights_ref_'+str(train_ref_ori)+'_sf_'+str(train_sf)+'_sep_'+str(train_sep)+'_lr_0.0001_trial_'+str(trial_num), init_conv4_weights.cpu())

            np.save(save_dir+'weights/conv5/init_conv5_weights_ref_'+str(train_ref_ori)+'_sf_'+str(train_sf)+'_sep_'+str(train_sep)+'_lr_0.0001_trial_'+str(trial_num), init_conv5_weights.cpu())

            # define the loss function and the optimizer
            criterion = nn.BCELoss()
            optimizer = optim.SGD(model[count].parameters(), lr=learning_rate, momentum=0.9)
            
            # If you want to use a cyclic learning rate
            if cyclicLR:
                scheduler = CyclicLR(optimizer, base_lr=learning_rate/10, max_lr=learning_rate*10, 
                        step_size_up=50, step_size_down=50, 
                        mode='triangular')

            # Get parameters
            train_root_dir = train_dir[0]
            train_ref_ori = train_dir[1]
            train_sf = train_dir[2]
            train_sep = train_dir[3]

            test_root_dir = test_dir[0]
            test_ref_ori = test_dir[1]
            test_sf = test_dir[2]
            test_sep = test_dir[3]

            seps.append(test_sep)

            # GET REFERENCE IMAGE FOR TRAINING
            train_ref_dir = 'StimulusImages/SG_refs/'
            train_ref_img = Image.open(train_ref_dir+'REFERENCE_ref_'+str(train_ref_ori)+'_sep_0.0_contr_1_ph_0.0_sf_'+str(train_sf)+'_NONE.png') # ref image for training set
            train_ref_img = data_transforms(train_ref_img).to(device)

            #plt.imshow(train_ref_img.permute(1,2,0))

            # load the dataset
            train_grating_dataset = GratingDataset(train_root_dir, transform=data_transforms, ref_orientation=ref_orientation, separation_angle=separation_angle, contrast=contrast, phase=phase, spatial_freq=spatial_freq, num_images=num_images)

            # create the dataloader
            train_dataloader = DataLoader(train_grating_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

            # GET REFERENCE IMAGE FOR TESTING
            test_ref_dir = 'StimulusImages/SG_refs/'
            test_ref_img = Image.open(test_ref_dir+'REFERENCE_ref_'+str(test_ref_ori)+'_sep_0.0_contr_1_ph_0.0_sf_'+str(test_sf)+'_NONE.png') # ref image for testing set
            test_ref_img = data_transforms(test_ref_img).to(device)

            #plt.imshow(test_ref_img.permute(1,2,0))

            # load the dataset
            test_grating_dataset = GratingDataset(test_root_dir, transform=data_transforms, ref_orientation=ref_orientation, separation_angle=separation_angle, contrast=contrast, phase=phase, spatial_freq=spatial_freq, num_images=num_images)

            # create the dataloader
            test_dataloader = DataLoader(test_grating_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
            
            transfer_dataset = GratingDataset(test_root_dir, transform=data_transforms, ref_orientation=ref_orientation, separation_angle=separation_angle, contrast=contrast, phase=phase, spatial_freq=spatial_freq, num_images=num_transfer_images)
            transfer_dataloader = DataLoader(transfer_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

            iter_count = 0
            weights_conv1 = []
            weights_conv2 = []
            weights_conv3 = []
            weights_conv4 = []
            weights_conv5 = []
            weights_fc = []

            summed_train_accuracies = []
            summed_test_accuracies = []
            train_accuracies = []
            test_accuracies = []

            # Compute individual gradient for a single sample
            def compute_grad(model, input, label, ref_img, sep):
                input = input.unsqueeze(0)  # prepend batch dimension for processing
                label = label.unsqueeze(0)

                prediction = model(input, ref_img, sep)
                loss = criterion(prediction[0], label)

                return torch.autograd.grad(loss, list(model.parameters()))

            # Compute individual gradients for entire batch
            def compute_sample_grads(model, inputs, labels, ref_img, sep):
                """ manually process each sample with per sample gradient """
                sample_grads = [compute_grad(model, inputs[i], labels[i], ref_img, sep) for i in range(batch_size)]#*2)]
                sample_grads = zip(*sample_grads)
                sample_grads = [torch.stack(shards) for shards in sample_grads]
                return sample_grads

            # train the model

            torch.cuda.empty_cache()
            
            for epoch in range(num_epochs):
                    # Without forced training
                    for (train_i, train_data), (test_i, test_data) in zip(enumerate(train_dataloader), enumerate(test_dataloader)):
                        # Force sequential/antisequential
                        if batch_size == 1 and step_count < num_steps and (forced_sequential or forced_antisequential):
                            train_dataloader.dataset.set_scaled(True)
                            train_dataloader.dataset.set_factor(factor)
                            if forced_sequential:
                                train_dataloader.dataset.set_force_type('seq')
                            elif forced_antisequential:
                                train_dataloader.dataset.set_force_type('antiseq')
                            train_data = next(iter(train_dataloader))
                        # No forced training
                        else:
                            train_dataloader.dataset.set_scaled(False)
                            train_dataloader.dataset.set_force_type('none')

                        # Single sample training
                        if batch_size == 1:
                            train_data = list([train_data[0], train_data[1], np.asarray(train_data[2]).flatten()])
                            test_data = list([test_data[0], test_data[1], np.asarray(test_data[2]).flatten()])
                        else:
                            train_data = list([torch.cat(torch.unbind(train_data[0], 1)), torch.cat(torch.unbind(train_data[1], 1)), np.asarray(train_data[2]).flatten()])
                            test_data = list([torch.cat(torch.unbind(test_data[0], 1)), torch.cat(torch.unbind(test_data[1], 1)), np.asarray(test_data[2]).flatten()])
                        
                        train_idx = torch.randperm(train_data[0].shape[0])
                        train_data[0] = train_data[0][train_idx].view(train_data[0].size())
                        train_data[1] = train_data[1][train_idx].view(train_data[1].size())
                        train_data[2] = train_data[2][train_idx]

                        test_idx = torch.randperm(test_data[0].shape[0])
                        test_data[0] = test_data[0][test_idx].view(test_data[0].size())
                        test_data[1] = test_data[1][test_idx].view(test_data[1].size())
                        test_data[2] = test_data[2][test_idx]

                        if iter_count % 20 == 0 or iter_count == 0:
                            # Regular weights:
                            c1 = model[count].conv1.weight.data
                            c1 = c1.detach().clone()
                            weights_conv1.append(c1.cpu())

                            c2 = model[count].conv2.weight.data
                            c2 = c2.detach().clone()
                            weights_conv2.append(c2.cpu())

                            c3 = model[count].conv3.weight.data
                            c3 = c3.detach().clone()
                            weights_conv3.append(c3.cpu())

                            c4 = model[count].conv4.weight.data
                            c4 = c4.detach().clone()
                            weights_conv4.append(c4.cpu())

                            c5 = model[count].conv5.weight.data
                            c5 = c5.detach().clone()
                            weights_conv5.append(c5.cpu())

                            # Skip connection weights:
                            fc = model[count].fc1.weight.data
                            fc = fc.detach().clone()
                            weights_fc.append(fc.cpu())
                        

                        ############ TRAINING ############

                        # set model to train mode
                        model[count].train()

                        train_inputs, train_labels, train_seps = train_data
                        train_inputs = train_inputs.to(device)
                        train_labels = train_labels.to(device)

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # plt.imshow(train_ref_img.cpu().permute(1,2,0))
                        # plt.savefig(save_dir+'sample_train_ref_img.svg')

                        train_outputs = model[count](train_inputs, train_ref_img, train_sep)
                        train_labels = train_labels.unsqueeze(1)
                        train_labels = train_labels.float()
                        train_loss = criterion(train_outputs[0], train_labels)

                        if collecting_gradients:# and dir_num == 0 and epoch == 0:
                            # Get per-sample gradients
                            per_sample_grads = compute_sample_grads(model[count], train_inputs, train_labels, train_ref_img, train_sep)
                            per_sample_grad_means = []
                            # for grad in per_sample_grads:
                            #     torch.mean(per_sample_grads, dim=1)

                            # Save per-sample gradients & samples that generated those gradients
                            torch.save(per_sample_grads, save_dir+'data/gradients_per_sample_iteration_'+str(step_count)+'_trial_'+str(trial_num)+'.pt')
                            #torch.save(per_sample_grad_means, save_dir='data/gradient_means_per_sample_iteration_'+str(iter_count)+'.pt')
                            # torch.save(train_inputs, save_dir+'data/samples_iteration_'+str(iter_count)+'.pt')
                            # torch.save(train_labels, save_dir+'data/labels_iteration_'+str(iter_count)+'.pt')
                        torch.save(train_seps, save_dir+'data/seps_iteration_'+str(step_count)+'_trial_'+str(trial_num)+'.pt')

                        # plt.imshow(train_inputs[idx_CW].cpu().permute(1,2,0))
                        # plt.savefig(save_dir+'sample_train_CW_img_5.0.svg')
                        # plt.imshow(train_inputs[idx_CCW].cpu().permute(1,2,0))
                        # plt.savefig(save_dir+'sample_train_CCW_img_5.0.svg')

                        # Backprop
                        train_loss.backward()

                        optimizer.step()
                        if cyclicLR and step_count >= num_steps:
                            scheduler.step()
                        
                        # Manually reset frozen readout weights if needed (so they don't change during training)
                        # This has to be done manually because we can't freeze only part of the readout layer with .requires_grad
                        if frozen_readout:
                            for i, x in zip(other_indices, fc_weights[0]):
                                model[count].fc1.weight.data[0][i] = x

                        with torch.no_grad():
                            # Add noise to confidence before calculating accuracy
                            confidence = train_outputs[0]
                            noisy_confidence = torch.add(confidence, torch.randn_like(confidence) * confidence_noise_sd)
                            # Clamp output to 0-1 (probability range)
                            noisy_confidence = torch.clamp(noisy_confidence, 0, 1)
                            train_confidence.append(torch.mean(torch.abs(confidence-0.5)*2).cpu())
                            train_noisy_confidence.append(torch.mean(torch.abs(noisy_confidence-0.5)*2).cpu())

                            # Calculate training accuracy for this iteration
                            output_labels = []
                            for item in noisy_confidence:
                                if item >= 0.5: output_labels.append(float(1))
                                else: output_labels.append(float(0))
                            train_acc = [output_labels[i] == train_labels[i] for i in range(0,len(output_labels))]
                            train_acc = [float(x) for x in train_acc]
                            train_acc = mean(train_acc)
                            train_accuracies.append(train_acc)


                            ############## TESTING ##############

                            model[count].eval()

                            test_inputs, test_labels, test_seps = test_data
                            test_inputs = test_inputs.to(device)
                            test_labels = test_labels.to(device)

                            test_outputs = model[count](test_inputs, test_ref_img, test_sep)
                            test_labels = test_labels.unsqueeze(1)
                            test_labels = test_labels.float()
                            test_loss = criterion(test_outputs[0], test_labels)
                            test_losses[dir_count].append(test_loss.item())

                            # plt.imshow(test_ref_img.cpu().permute(1,2,0))
                            # plt.savefig(save_dir+'sample_test_ref_img.svg')

                            # idx_CW = np.argmax(test_labels.cpu())
                            # idx_CCW = np.argmin(test_labels.cpu())

                            # plt.imshow(test_inputs[idx_CW].cpu().permute(1,2,0))
                            # plt.savefig(save_dir+'sample_test_CW_img_1.0.svg')
                            # plt.imshow(test_inputs[idx_CCW].cpu().permute(1,2,0))
                            # plt.savefig(save_dir+'sample_test_CCW_img_1.0.svg')

                            # Add noise to confidence before calculating accuracy
                            confidence = test_outputs[0]
                            noisy_confidence = torch.add(confidence, torch.randn_like(confidence) * confidence_noise_sd)
                            # Clamp output to 0-1 (probability range)
                            noisy_confidence = torch.clamp(noisy_confidence, 0, 1)
                            test_confidence.append(torch.mean(torch.abs(confidence-0.5)*2).cpu())
                            test_noisy_confidence.append(torch.mean(torch.abs(noisy_confidence-0.5)*2).cpu())

                            # Calculate testing accuracy for this iteration
                            output_labels = []
                            for item in noisy_confidence:
                                if item >= 0.5: output_labels.append(float(1))
                                else: output_labels.append(float(0))
                            test_acc = [output_labels[i] == test_labels[i] for i in range(0,len(output_labels))]
                            test_acc = [float(x) for x in test_acc]
                            test_acc = mean(test_acc)
                            test_accuracies.append(test_acc)

                            test_summed_acc = mean(test_accuracies)
                            train_summed_acc = mean(train_accuracies)

                            summed_test_accuracies.append(test_summed_acc)
                            summed_train_accuracies.append(train_summed_acc)

                            # print statistics
                            print('[%d, %5d] loss: %.3f, %.3f accuracy: %.3f, %.3f confidence: %.3f, %.3f noisy confidence: %.3f, %.3f' % (epoch + 1, train_i + 1, train_loss.item(), test_loss.item(), train_accuracies[iter_count], test_accuracies[iter_count], train_confidence[-1], test_confidence[-1], train_noisy_confidence[-1], test_noisy_confidence[-1]))

                        iter_count += 1
                        step_count += 1
            
            print('Finished Training & Testing')

            # store final weights for each model

            param_list = [*model[count].parameters()]
        
            # Regular weights:
            final_conv1_weights = model[count].conv1.weight.data
            final_conv1_weights = final_conv1_weights.detach().clone()

            final_conv2_weights = model[count].conv2.weight.data
            final_conv2_weights = final_conv2_weights.detach().clone()

            final_conv3_weights = model[count].conv3.weight.data
            final_conv3_weights = final_conv3_weights.detach().clone()

            final_conv4_weights = model[count].conv4.weight.data
            final_conv4_weights = final_conv4_weights.detach().clone()

            final_conv5_weights = model[count].conv5.weight.data
            final_conv5_weights = final_conv5_weights.detach().clone()

            # Skip connection weights:
            final_fc_weights = model[count].fc1.weight.data
            final_fc_weights = final_fc_weights.detach().clone()

            # SAVE MODEL, DATA & WEIGHTS
            # Save model
            torch.save(model[count].state_dict(), save_dir+'models/original_model_'+str(train_ref_ori)+'_sf_'+str(train_sf)+'_sep_'+str(train_sep)+'_trial_'+str(trial_num)+'.pth')

            # Save loss data
            np.savetxt(save_dir+'data/train_data_ref_'+str(train_ref_ori)+'_sf_'+str(train_sf)+'_sep_'+str(train_sep)+'_lr_0.0001_trial_'+str(trial_num)+'.csv', train_losses[dir_count], delimiter=",")
            np.savetxt(save_dir+'data/test_data_ref_'+str(test_ref_ori)+'_sf_'+str(test_sf)+'_sep_'+str(test_sep)+'_lr_0.0001_trial_'+str(trial_num)+'.csv', test_losses[dir_count], delimiter=",")

            # Save accuracy data
            np.savetxt(save_dir+'data/train_accuracy_ref_'+str(train_ref_ori)+'_sf_'+str(train_sf)+'_sep_'+str(train_sep)+'_lr_0.0001_trial_'+str(trial_num)+'.csv', train_accuracies, delimiter=",")
            np.savetxt(save_dir+'data/test_accuracy_ref_'+str(test_ref_ori)+'_sf_'+str(test_sf)+'_sep_'+str(test_sep)+'_lr_0.0001_trial_'+str(trial_num)+'.csv', test_accuracies, delimiter=",")

            # Save confidence data
            np.savetxt(save_dir+'data/train_noisy_confidence_ref_'+str(train_ref_ori)+'_sf_'+str(train_sf)+'_sep_'+str(train_sep)+'_lr_0.0001_trial_'+str(trial_num)+'.csv', train_noisy_confidence, delimiter=",")
            np.savetxt(save_dir+'data/test_noisy_confidence_ref_'+str(test_ref_ori)+'_sf_'+str(test_sf)+'_sep_'+str(test_sep)+'_lr_0.0001_trial_'+str(trial_num)+'.csv', test_noisy_confidence, delimiter=",")
            np.savetxt(save_dir+'data/train_confidence_ref_'+str(train_ref_ori)+'_sf_'+str(train_sf)+'_sep_'+str(train_sep)+'_lr_0.0001_trial_'+str(trial_num)+'.csv', train_confidence, delimiter=",")
            np.savetxt(save_dir+'data/test_confidence_ref_'+str(test_ref_ori)+'_sf_'+str(test_sf)+'_sep_'+str(test_sep)+'_lr_0.0001_trial_'+str(trial_num)+'.csv', test_confidence, delimiter=",")

            # Save summed accuracy data
            np.savetxt(save_dir+'data/summed_train_accuracy_ref_'+str(train_ref_ori)+'_sf_'+str(train_sf)+'_sep_'+str(train_sep)+'_lr_0.0001_trial_'+str(trial_num)+'.csv', summed_train_accuracies, delimiter=",")
            np.savetxt(save_dir+'data/summed_test_accuracy_ref_'+str(test_ref_ori)+'_sf_'+str(test_sf)+'_sep_'+str(test_sep)+'_lr_0.0001_trial_'+str(trial_num)+'.csv', summed_test_accuracies, delimiter=",")

            # Save skip weights
            np.savetxt(save_dir+'weights/final_fc_weights_ref_'+str(train_ref_ori)+'_sf_'+str(train_sf)+'_sep_'+str(train_sep)+'_lr_0.0001_trial_'+str(trial_num)+'.csv', final_fc_weights.cpu(), delimiter=",")

            # Save regular weights
            np.save(save_dir+'weights/final_conv1_weights_ref_'+str(train_ref_ori)+'_sf_'+str(train_sf)+'_sep_'+str(train_sep)+'_lr_0.0001_trial_'+str(trial_num), final_conv1_weights.cpu())
            np.save(save_dir+'weights/final_conv2_weights_ref_'+str(train_ref_ori)+'_sf_'+str(train_sf)+'_sep_'+str(train_sep)+'_lr_0.0001_trial_'+str(trial_num), final_conv2_weights.cpu())
            np.save(save_dir+'weights/final_conv3_weights_ref_'+str(train_ref_ori)+'_sf_'+str(train_sf)+'_sep_'+str(train_sep)+'_lr_0.0001_trial_'+str(trial_num), final_conv3_weights.cpu())
            np.save(save_dir+'weights/final_conv4_weights_ref_'+str(train_ref_ori)+'_sf_'+str(train_sf)+'_sep_'+str(train_sep)+'_lr_0.0001_trial_'+str(trial_num), final_conv4_weights.cpu())
            np.save(save_dir+'weights/final_conv5_weights_ref_'+str(train_ref_ori)+'_sf_'+str(train_sf)+'_sep_'+str(train_sep)+'_lr_0.0001_trial_'+str(trial_num), final_conv5_weights.cpu())

            # save weights throughout training
            weights_conv1 = np.asanyarray(weights_conv1, dtype=object)
            weights_conv2 = np.asanyarray(weights_conv2, dtype=object)
            weights_conv3 = np.asanyarray(weights_conv3, dtype=object)
            weights_conv4 = np.asanyarray(weights_conv4, dtype=object)
            weights_conv5 = np.asanyarray(weights_conv5, dtype=object)

            weights_fc = np.asanyarray(weights_fc, dtype=object)

            np.save(save_dir+'weights/all_conv1_weights_ref_'+str(train_ref_ori)+'_sf_'+str(train_sf)+'_sep_'+str(train_sep)+'_lr_0.0001_trial_'+str(trial_num), weights_conv1, allow_pickle=True)
            np.save(save_dir+'weights/all_conv2_weights_ref_'+str(train_ref_ori)+'_sf_'+str(train_sf)+'_sep_'+str(train_sep)+'_lr_0.0001_trial_'+str(trial_num), weights_conv2, allow_pickle=True)
            np.save(save_dir+'weights/all_conv3_weights_ref_'+str(train_ref_ori)+'_sf_'+str(train_sf)+'_sep_'+str(train_sep)+'_lr_0.0001_trial_'+str(trial_num), weights_conv3, allow_pickle=True)
            np.save(save_dir+'weights/all_conv4_weights_ref_'+str(train_ref_ori)+'_sf_'+str(train_sf)+'_sep_'+str(train_sep)+'_lr_0.0001_trial_'+str(trial_num), weights_conv4, allow_pickle=True)
            np.save(save_dir+'weights/all_conv5_weights_ref_'+str(train_ref_ori)+'_sf_'+str(train_sf)+'_sep_'+str(train_sep)+'_lr_0.0001_trial_'+str(trial_num), weights_conv5, allow_pickle=True)

            np.save(save_dir+'weights/all_fc_weights_ref_'+str(train_ref_ori)+'_sf_'+str(train_sf)+'_sep_'+str(train_sep)+'_lr_0.0001_trial_'+str(trial_num), weights_fc, allow_pickle=True)
            
            transfer_losses = []
            transfer_accuracies = []
            transfer_confidence = []
            transfer_noisy_confidence = []
            
            with torch.no_grad():
                for (test_i, test_data) in enumerate(transfer_dataloader):
                    if batch_size == 1:
                        test_data = list([test_data[0], test_data[1]])
                    else:
                        test_data = list([torch.cat(torch.unbind(test_data[0], 1)), torch.cat(torch.unbind(test_data[1], 1))])
                    
                    test_idx = torch.randperm(test_data[0].shape[0])
                    test_data[0] = test_data[0][test_idx].view(test_data[0].size())
                    test_data[1] = test_data[1][test_idx].view(test_data[1].size())

                    ############## TRANSFER ##############

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

                    # print statistics
                    print('[%d, %5d] loss: %.3f accuracy: %.3f confidence: %.3f noisy confidence: %.3f' % (epoch + 1, train_i + 1, test_loss.item(), test_acc, transfer_confidence[-1], transfer_noisy_confidence[-1]))

                np.savetxt(save_dir+'data/transfer_data_ref_'+str(test_ref_ori)+'_sf_'+str(test_sf)+'_sep_'+str(test_sep)+'_lr_0.0001_trial_'+str(trial_num)+'.csv', transfer_losses, delimiter=",")
                np.savetxt(save_dir+'data/transfer_accuracy_ref_'+str(test_ref_ori)+'_sf_'+str(test_sf)+'_sep_'+str(test_sep)+'_lr_0.0001_trial_'+str(trial_num)+'.csv', transfer_accuracies, delimiter=",")
                np.savetxt(save_dir+'data/transfer_confidence_ref_'+str(test_ref_ori)+'_sf_'+str(test_sf)+'_sep_'+str(test_sep)+'_lr_0.0001_trial_'+str(trial_num)+'.csv', transfer_confidence, delimiter=",")
                np.savetxt(save_dir+'data/transfer_noisy_confidence_ref_'+str(test_ref_ori)+'_sf_'+str(test_sf)+'_sep_'+str(test_sep)+'_lr_0.0001_trial_'+str(trial_num)+'.csv', transfer_noisy_confidence, delimiter=",")
                print('Finished Testing Transfer')
                
                if not sequential:
                    count += 1
                test_count += 1

                dir_count += 1

