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
from statistics import mean
import os
from copy_weights import copy_weights_alexnet, copy_weights_resnet18, copy_weights_vgg16, copy_weights_vgg11, copy_weights_densenet121, copy_weights_convnext
from copy_weights import copy_weights_squeezenet, copy_weights_mnasnet, copy_weights_mobilenetv2, copy_weights_mobilenetv3, copy_weights_efficientnet, copy_weights_googlenet

if __name__ == '__main__':
# for factor in [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # set the parameters for the model
    num_classes = 2
    num_epochs = 5
    learning_rate = 0.0001

    # set the parameters for the dataset
    batch_size = 1 # controls single-sample or multi-sample update - for multi-sample, batch size is actually this value * 2 (equal numbers of CW and CCW images)

    # Number of models to train
    num_trials = 20
    
    # set the parameters for training
    sequential = True
    # model_name = os.environ.get('MODEL_NAME')
    model_name = 'AlexNet'
    transfer_condition = 'SF' # or: 'ref_ori'
    pretrained = False
    collecting_gradients = False
    cyclicLR = False
    forced_sequential = False
    forced_antisequential = False
    factor = 1.0
    num_steps = 25
    frozen_readout = False
    random_readout = False
    frozen_conv = False

    # Replace with the directory where you want to save the outputs
    base_dir = '../saved_outputs/shuffled_curriculum/'

    # Where to load readout weight subspace to train model with frozen readout from
    if frozen_readout:
        frozen_source_model_dir = 'saved_outputs/shuffled_forced_sequential_factor_6.0/' # set here

    num_transfer_images = 200
    if batch_size == 1: num_images = 100
    elif batch_size == 20: num_images = 1000
    else: num_images == batch_size * 100 // 2

    trials = range(1,num_trials+1)

    if transfer_condition == 'ref_ori':
        root_dir = 'StimulusImages/SG_train_double_sf/'
        test_root_dir = 'StimulusImages/SG_test_changed_ref_15/'
        spatial_freq_train = 0.05
        spatial_freq_test = 0.05
        ref_angle_train = 0
        ref_angle_test = 15
    elif transfer_condition == 'SF':
        root_dir = 'StimulusImages/SG_train_double_sf/'
        test_root_dir = 'StimulusImages/SG_test_double_sf/'
        spatial_freq_train = 0.05
        spatial_freq_test = 0.1
        ref_angle_train = 0
        ref_angle_test = 0

    # format: [[train/test]root_dir + [dir with correct sep angle], ref orientation, spatial freq, sep_angle]
    dir_list = [
                # [root_dir+'sep_10.0',ref_angle_train,spatial_freq_train,10.0],
                # [root_dir+'sep_5.0',ref_angle_train,spatial_freq_train,5.0],
                # [root_dir+'sep_2.0',ref_angle_train,spatial_freq_train,2.0],
                # [root_dir+'sep_1.0',ref_angle_train,spatial_freq_train,1.0],
                # [root_dir+'sep_0.5',ref_angle_train,spatial_freq_train,0.5],
                [root_dir+'sep_5.0_1.0_combined',ref_angle_train,spatial_freq_train,5.0],
                [root_dir+'sep_5.0_1.0_combined',ref_angle_train,spatial_freq_train,1.0],
                # [root_dir+'sep_5.0',ref_angle_train,spatial_freq_train,5.0],
                # [root_dir+'sep_1.0',ref_angle_train,spatial_freq_train,1.0],
                # [root_dir+'sep_1.0',ref_angle_train,spatial_freq_train,5.0],
                # [root_dir+'sep_1.0',ref_angle_train,spatial_freq_train,1.0],
                ]
    
    test_dir_list = [
                    # [test_root_dir+'sep_10.0',ref_angle_test,spatial_freq_test,10.0],
                    # [test_root_dir+'sep_5.0',ref_angle_test,spatial_freq_test,5.0],
                    # [test_root_dir+'sep_2.0',ref_angle_test,spatial_freq_test,2.0],
                    # [test_root_dir+'sep_1.0',ref_angle_test,spatial_freq_test,1.0],
                    # [test_root_dir+'sep_0.5',ref_angle_test,spatial_freq_test,0.5],
                    [test_root_dir+'sep_1.0',ref_angle_test,spatial_freq_test,5.0],
                    [test_root_dir+'sep_1.0',ref_angle_test,spatial_freq_test,1.0],
                    ]

    if model_name == 'ResNet18_skip':
        from models.ResNet18_skip import ResNet18 as ResNet
    elif model_name == 'VGG16_skip':
        from models.VGG16_skip import VGG16
    elif model_name == 'VGG11_skip':
        from models.VGG11_skip import VGG11
    elif model_name == 'DenseNet121_skip':
        from models.DenseNet121_skip import DenseNet121
    elif model_name == 'ConvNeXt_skip':
        from models.ConvNeXt_skip import ConvNeXt
    elif model_name == 'SqueezeNet_skip':
        from models.SqueezeNet_skip import SqueezeNet
    elif model_name == 'MNASNet_skip':
        from models.MNASNet_skip import MNASNet
    elif model_name == 'MobileNetV2_skip':
        from models.MobileNetV2_skip import MobileNetV2
    elif model_name == 'MobileNetV3_skip':
        from models.MobileNetV3_skip import MobileNetV3
    elif model_name == 'EfficientNet_skip':
        from models.EfficientNet_skip import EfficientNet
    elif model_name == 'GoogLeNet_skip':
        from models.GoogLeNet_skip import GoogLeNet
    elif model_name == 'AlexNet' or model_name == 'AlexNet_skip':
        if random_readout:
            from models.skip_alexnet_random_init import AlexNet
        else:
            from models.skip_alexnet import AlexNet
    else:
        raise ValueError('Invalid model name')

    if batch_size == 1:
        from dataset_single_example import GratingDataset
    else:
        from dataset_shuffled import GratingDataset

    # set the transforms
    stimulus_noise_sd = 0.02 #25/255 # Caffe does noise in pixel values, torch does it in 0-1 tensor space
    confidence_noise_sd = 0.3 #25/255

    data_transforms = transforms.Compose([
        transforms.Resize(227),
        transforms.ToTensor(),
        GaussianNoise(0, stimulus_noise_sd), # STANDARD DEVIATION OF GAUSSIAN NOISE
    ])
    
    if forced_sequential: forced_name = 'sequential'
    if forced_antisequential: forced_name = 'antisequential'

    if sequential:
        save_dir = base_dir
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        if not os.path.exists(save_dir+'models/'): os.makedirs(save_dir+'models/')
        if not os.path.exists(save_dir+'data/'): os.makedirs(save_dir+'data/')
        if not os.path.exists(save_dir+'weights/'): os.makedirs(save_dir+'weights/')

    for trial_num in trials:

        test_count = 0
        count = 0

        model = []
        seps = []

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
                if model_name == 'AlexNet' or model_name == 'AlexNet_skip':
                    model.append(AlexNet())
                    alexnet = torchvision.models.alexnet(pretrained=True)
                    copy_weights_alexnet(model[count], alexnet)
                elif model_name == 'ResNet18_skip':
                    model.append(ResNet())
                    resnet = torchvision.models.resnet18(pretrained=True)
                    copy_weights_resnet18(model[count], resnet)
                elif model_name == 'VGG16_skip':
                    model.append(VGG16())
                    vgg = torchvision.models.vgg16(pretrained=True)
                    copy_weights_vgg16(model[count], vgg)
                elif model_name == 'VGG11_skip':
                    model.append(VGG11())
                    vgg = torchvision.models.vgg11(pretrained=True)
                    copy_weights_vgg11(model[count], vgg)
                elif model_name == 'DenseNet121_skip':
                    model.append(DenseNet121())
                    densenet = torchvision.models.densenet121(pretrained=True)
                    copy_weights_densenet121(model[count], densenet)
                elif model_name == 'ConvNeXt_skip':
                    model.append(ConvNeXt())
                    convnext = torchvision.models.convnext_tiny(pretrained=True)
                    copy_weights_convnext(model[count], convnext)
                elif model_name == 'SqueezeNet_skip':
                    model.append(SqueezeNet())
                    squeezenet = torchvision.models.squeezenet1_1(pretrained=True)
                    copy_weights_squeezenet(model[count], squeezenet)
                elif model_name == 'MNASNet_skip':
                    model.append(MNASNet())
                    mnasnet = torchvision.models.mnasnet1_0(pretrained=True)
                    copy_weights_mnasnet(model[count], mnasnet)
                elif model_name == 'MobileNetV2_skip':
                    model.append(MobileNetV2())
                    mobilenetv2 = torchvision.models.mobilenet_v2(pretrained=True)
                    copy_weights_mobilenetv2(model[count], mobilenetv2)
                elif model_name == 'MobileNetV3_skip':
                    model.append(MobileNetV3())
                    mobilenetv3 = torchvision.models.mobilenet_v3_small(pretrained=True)
                    copy_weights_mobilenetv3(model[count], mobilenetv3)
                elif model_name == 'EfficientNet_skip':
                    model.append(EfficientNet())
                    efficientnet = torchvision.models.efficientnet_b0(pretrained=True)
                    copy_weights_efficientnet(model[count], efficientnet)
                elif model_name == 'GoogLeNet_skip':
                    model.append(GoogLeNet())
                    googlenet = torchvision.models.googlenet(
                        weights=torchvision.models.GoogLeNet_Weights.IMAGENET1K_V1,
                        aux_logits=True,
                    )
                    copy_weights_googlenet(model[count], googlenet)
                model[count] = model[count].to(device)
            
            else:
                # For loading old models - If you want to use a pretrained model
                model_dir = save_dir+'old_model_dir/'
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
            
            # Clear GPU cache at the start of each curriculum stage
            torch.cuda.empty_cache()
            
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
                if model_name == 'AlexNet' or model_name == 'AlexNet_skip':
                    model.append(AlexNet())
                    alexnet = torchvision.models.alexnet(pretrained=True)
                    copy_weights_alexnet(model[count], alexnet)
                elif model_name == 'ResNet18_skip':
                    model.append(ResNet())
                    resnet = torchvision.models.resnet18(pretrained=True)
                    copy_weights_resnet18(model[count], resnet)
                elif model_name == 'VGG16_skip':
                    model.append(VGG16())
                    vgg = torchvision.models.vgg16(pretrained=True)
                    copy_weights_vgg16(model[count], vgg)
                elif model_name == 'VGG11_skip':
                    model.append(VGG11())
                    vgg = torchvision.models.vgg11(pretrained=True)
                    copy_weights_vgg11(model[count], vgg)
                elif model_name == 'DenseNet121_skip':
                    model.append(DenseNet121())
                    densenet = torchvision.models.densenet121(pretrained=True)
                    copy_weights_densenet121(model[count], densenet)
                elif model_name == 'ConvNeXt_skip':
                    model.append(ConvNeXt())
                    convnext = torchvision.models.convnext_tiny(pretrained=True)
                    copy_weights_convnext(model[count], convnext)
                elif model_name == 'SqueezeNet_skip':
                    model.append(SqueezeNet())
                    squeezenet = torchvision.models.squeezenet1_1(pretrained=True)
                    copy_weights_squeezenet(model[count], squeezenet)
                elif model_name == 'MNASNet_skip':
                    model.append(MNASNet())
                    mnasnet = torchvision.models.mnasnet1_0(pretrained=True)
                    copy_weights_mnasnet(model[count], mnasnet)
                elif model_name == 'MobileNetV2_skip':
                    model.append(MobileNetV2())
                    mobilenetv2 = torchvision.models.mobilenet_v2(pretrained=True)
                    copy_weights_mobilenetv2(model[count], mobilenetv2)
                elif model_name == 'MobileNetV3_skip':
                    model.append(MobileNetV3())
                    mobilenetv3 = torchvision.models.mobilenet_v3_small(pretrained=True)
                    copy_weights_mobilenetv3(model[count], mobilenetv3)
                elif model_name == 'EfficientNet_skip':
                    model.append(EfficientNet())
                    efficientnet = torchvision.models.efficientnet_b0(pretrained=True)
                    copy_weights_efficientnet(model[count], efficientnet)
                elif model_name == 'GoogLeNet_skip':
                    model.append(GoogLeNet())
                    googlenet = torchvision.models.googlenet(
                        weights=torchvision.models.GoogLeNet_Weights.IMAGENET1K_V1,
                        aux_logits=True,
                    )
                    copy_weights_googlenet(model[count], googlenet)
                model[count] = model[count].to(device)

            if frozen_conv:
                for param in model[count].parameters():
                    param.requires_grad = False
                
                if model_name == "AlexNet" or model_name == "AlexNet_skip" or model_name == "VGG11_skip" or model_name == "VGG16_skip":
                    model[count].fc1.weight.requires_grad = True
                    model[count].fc1.bias.requires_grad = True
                else:
                    try:
                        model[count].fc.weight.requires_grad = True
                        model[count].fc.bias.requires_grad = True
                    except:
                        print("Model fully connected layer not named fc")

            # store initial weights
            if model_name == 'AlexNet' or model_name == 'AlexNet_skip': # Save all conv layer weights for AlexNet since it's a smaller model
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

            elif model_name == 'VGG11_skip' or model_name == 'VGG16_skip': # Just save last FC Layer weights
                init_fc_weights = model[count].fc1.weight.data
            else:
                try:
                    init_fc_weights = model[count].fc.weight.data
                except:
                    print("Model fully connected layer not named fc")
                    
            init_fc_weights = init_fc_weights.detach().clone()


            # Save initial weights

            train_ref_ori = train_dir[1]
            train_sf = train_dir[2]
            train_sep = train_dir[3]

            if not sequential:
                save_dir = base_dir +str(train_sep)+'_angle_sep/'
                if not os.path.exists(save_dir): os.makedirs(save_dir)
                if not os.path.exists(save_dir+'models/'): os.makedirs(save_dir+'models/')
                if not os.path.exists(save_dir+'data/'): os.makedirs(save_dir+'data/')
                if not os.path.exists(save_dir+'weights/'): os.makedirs(save_dir+'weights/')

            # Save top 150 highest magnitude initial FC weights (memory efficient)
            init_fc_flat = init_fc_weights.view(-1)
            init_top_values, init_top_indices = torch.topk(init_fc_flat.abs(), k=min(150, init_fc_flat.numel()))
            init_top_weights_with_sign = init_fc_flat[init_top_indices]
            np.savetxt(save_dir+'weights/top150_init_fc_indices_ref_'+str(train_ref_ori)+'_sf_'+str(train_sf)+'_sep_'+str(train_sep)+'_lr_0.0001_trial_'+str(trial_num)+'.csv', init_top_indices.cpu(), delimiter=",")
            np.savetxt(save_dir+'weights/top150_init_fc_weights_ref_'+str(train_ref_ori)+'_sf_'+str(train_sf)+'_sep_'+str(train_sep)+'_lr_0.0001_trial_'+str(trial_num)+'.csv', init_top_weights_with_sign.cpu(), delimiter=",")
            del init_fc_flat, init_top_values, init_top_indices, init_top_weights_with_sign

            if model_name == 'AlexNet' or model_name == 'AlexNet_skip':
                # Save regular weights
                np.save(save_dir+'weights/init_conv1_weights_ref_'+str(train_ref_ori)+'_sf_'+str(train_sf)+'_sep_'+str(train_sep)+'_lr_0.0001_trial_'+str(trial_num), init_conv1_weights.cpu())
                np.save(save_dir+'weights/init_conv2_weights_ref_'+str(train_ref_ori)+'_sf_'+str(train_sf)+'_sep_'+str(train_sep)+'_lr_0.0001_trial_'+str(trial_num), init_conv2_weights.cpu())
                np.save(save_dir+'weights/init_conv3_weights_ref_'+str(train_ref_ori)+'_sf_'+str(train_sf)+'_sep_'+str(train_sep)+'_lr_0.0001_trial_'+str(trial_num), init_conv3_weights.cpu())
                np.save(save_dir+'weights/init_conv4_weights_ref_'+str(train_ref_ori)+'_sf_'+str(train_sf)+'_sep_'+str(train_sep)+'_lr_0.0001_trial_'+str(trial_num), init_conv4_weights.cpu())
                np.save(save_dir+'weights/init_conv5_weights_ref_'+str(train_ref_ori)+'_sf_'+str(train_sf)+'_sep_'+str(train_sep)+'_lr_0.0001_trial_'+str(trial_num), init_conv5_weights.cpu())

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
            print("Train ref img name: " + train_ref_dir+'REFERENCE_ref_'+str(train_ref_ori)+'_sep_0.0_contr_1_ph_0.0_sf_'+str(train_sf)+'_NONE.png')
            train_ref_img = data_transforms(train_ref_img).to(device)

            # plt.imshow(train_ref_img.permute(1,2,0))

            # load the dataset
            train_grating_dataset = GratingDataset(train_root_dir, transform=data_transforms, num_images=num_images)

            # create the dataloader
            train_dataloader = DataLoader(train_grating_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
            
            if collecting_gradients:
                # Create a separate dataloader for fetching alternate angle samples (won't interfere with main loop)
                alternate_dataloader = DataLoader(train_grating_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
                alternate_iter = iter(alternate_dataloader)

            # GET REFERENCE IMAGE FOR TESTING
            test_ref_dir = 'StimulusImages/SG_refs/'
            test_ref_img = Image.open(test_ref_dir+'REFERENCE_ref_'+str(test_ref_ori)+'_sep_0.0_contr_1_ph_0.0_sf_'+str(test_sf)+'_NONE.png') # ref image for testing set
            print("Test ref img name: " + test_ref_dir+'REFERENCE_ref_'+str(test_ref_ori)+'_sep_0.0_contr_1_ph_0.0_sf_'+str(test_sf)+'_NONE.png')
            test_ref_img = data_transforms(test_ref_img).to(device)

            # plt.imshow(test_ref_img.permute(1,2,0))

            # load the dataset
            test_grating_dataset = GratingDataset(test_root_dir, transform=data_transforms, num_images=num_images)

            # create the dataloader
            test_dataloader = DataLoader(test_grating_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
            
            transfer_dataset = GratingDataset(test_root_dir, transform=data_transforms, num_images=num_transfer_images)
            transfer_dataloader = DataLoader(transfer_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

            iter_count = 0
            if model_name == 'AlexNet' or model_name == 'AlexNet_skip':
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
                # manually process each sample with per sample gradient
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
                        if step_count < num_steps and (forced_sequential or forced_antisequential):
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

                        if (model_name == 'AlexNet' or model_name == 'AlexNet_skip') and (iter_count % 20 == 0 or iter_count == 0):
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

                        elif (model_name == 'VGG11_skip' or model_name == 'VGG16_skip') and (iter_count % 20 == 0 or iter_count == 0):
                            fc = model[count].fc1.weight.data
                        
                        elif (iter_count % 20 == 0 or iter_count == 0):
                            try:
                                fc = model[count].fc.weight.data
                            except:
                                print("Model fully connected layer not named fc")

                        # Save only top 150 highest magnitude weights (memory efficient)
                        fc_flat = fc.view(-1)
                        top_values, top_indices = torch.topk(fc_flat.abs(), k=min(150, fc_flat.numel()))
                        top_weights_with_sign = fc_flat[top_indices]  # Preserve sign
                        weights_fc.append({
                            'indices': top_indices.cpu(),
                            'values': top_weights_with_sign.cpu(),
                            'iteration': iter_count
                        })
                        

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
                            current_sep = float(train_seps[0])
                            per_sample_grads = compute_sample_grads(model[count], train_inputs, train_labels, train_ref_img, current_sep)
                            per_sample_grad_means = []
                            # for grad in per_sample_grads:
                            #     torch.mean(per_sample_grads, dim=1)

                            # Save per-sample gradients & samples that generated those gradients
                            torch.save(per_sample_grads, save_dir+'data/gradients_per_sample_iteration_'+str(step_count)+'_sep_'+str(current_sep)+'_trial_'+str(trial_num)+'.pt')
                            #torch.save(per_sample_grad_means, save_dir='data/gradient_means_per_sample_iteration_'+str(iter_count)+'.pt')
                            # torch.save(train_inputs, save_dir+'data/samples_iteration_'+str(iter_count)+'.pt')
                            # torch.save(train_labels, save_dir+'data/labels_iteration_'+str(iter_count)+'.pt')
                            
                            # Compute gradient for the alternate angle separation (without updating)
                            if batch_size == 1:
                                alternate_sep = float(1.0) if current_sep == 5.0 else float(5.0)
                                # Get a sample with the alternate separation
                                alternate_data = None
                                while alternate_data is None:
                                    try:
                                        temp_data = next(alternate_iter)
                                    except StopIteration:
                                        # Reset iterator if we've exhausted the dataloader
                                        alternate_iter = iter(alternate_dataloader)
                                        temp_data = next(alternate_iter)
                                    temp_sep = float(np.asarray(temp_data[2]).flatten()[0])
                                    if temp_sep == alternate_sep:
                                        alternate_data = temp_data
                                        break
                                
                                if alternate_data is not None:
                                    alt_inputs = alternate_data[0].to(device)
                                    alt_labels = alternate_data[1].to(device)
                                    alt_labels = alt_labels.unsqueeze(1)
                                    alt_labels = alt_labels.float()
                                    alt_seps = np.asarray(alternate_data[2]).flatten()
                                    # Compute gradient for alternate sample (but don't update)
                                    per_sample_grads_alternate = compute_sample_grads(model[count], alt_inputs, alt_labels, train_ref_img, alternate_sep)
                                    # Save alternate gradients
                                    torch.save(per_sample_grads_alternate, save_dir+'data/gradients_per_sample_iteration_'+str(step_count)+'_sep_'+str(alternate_sep)+'_trial_'+str(trial_num)+'.pt')
                                    # torch.save(alt_seps, save_dir+'data/seps_alternate_iteration_'+str(step_count)+'_trial_'+str(trial_num)+'.pt')
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
                            train_acc = [output_labels[i] == train_labels[i].item() for i in range(0,len(output_labels))]
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
                            test_acc = [output_labels[i] == test_labels[i].item() for i in range(0,len(output_labels))]
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

            # store final weights for each model (AlexNet only for now)
            if model_name == 'AlexNet':
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

            elif model_name == 'VGG11_skip' or model_name == 'VGG16_skip': # Just save last FC Layer weights
                final_fc_weights = model[count].fc1.weight.data
            else:
                try:
                    final_fc_weights = model[count].fc.weight.data
                except:
                    print("Model fully connected layer not named fc")
            
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

            # Save top 150 highest magnitude final FC weights (memory efficient)
            final_fc_flat = final_fc_weights.view(-1)
            final_top_values, final_top_indices = torch.topk(final_fc_flat.abs(), k=min(150, final_fc_flat.numel()))
            final_top_weights_with_sign = final_fc_flat[final_top_indices]
            np.savetxt(save_dir+'weights/top150_final_fc_indices_ref_'+str(train_ref_ori)+'_sf_'+str(train_sf)+'_sep_'+str(train_sep)+'_lr_0.0001_trial_'+str(trial_num)+'.csv', final_top_indices.cpu(), delimiter=",")
            np.savetxt(save_dir+'weights/top150_final_fc_weights_ref_'+str(train_ref_ori)+'_sf_'+str(train_sf)+'_sep_'+str(train_sep)+'_lr_0.0001_trial_'+str(trial_num)+'.csv', final_top_weights_with_sign.cpu(), delimiter=",")
            del final_fc_flat, final_top_values, final_top_indices, final_top_weights_with_sign

            # Save top 150 FC weights throughout training (memory efficient - saves indices & values only)
            weights_fc = np.asanyarray(weights_fc, dtype=object)
            np.save(save_dir+'weights/top150_fc_weights_ref_'+str(train_ref_ori)+'_sf_'+str(train_sf)+'_sep_'+str(train_sep)+'_lr_0.0001_trial_'+str(trial_num), weights_fc, allow_pickle=True)

            if model_name == 'AlexNet' or model_name == 'AlexNet_skip':
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

                np.save(save_dir+'weights/all_conv1_weights_ref_'+str(train_ref_ori)+'_sf_'+str(train_sf)+'_sep_'+str(train_sep)+'_lr_0.0001_trial_'+str(trial_num), weights_conv1, allow_pickle=True)
                np.save(save_dir+'weights/all_conv2_weights_ref_'+str(train_ref_ori)+'_sf_'+str(train_sf)+'_sep_'+str(train_sep)+'_lr_0.0001_trial_'+str(trial_num), weights_conv2, allow_pickle=True)
                np.save(save_dir+'weights/all_conv3_weights_ref_'+str(train_ref_ori)+'_sf_'+str(train_sf)+'_sep_'+str(train_sep)+'_lr_0.0001_trial_'+str(trial_num), weights_conv3, allow_pickle=True)
                np.save(save_dir+'weights/all_conv4_weights_ref_'+str(train_ref_ori)+'_sf_'+str(train_sf)+'_sep_'+str(train_sep)+'_lr_0.0001_trial_'+str(trial_num), weights_conv4, allow_pickle=True)
                np.save(save_dir+'weights/all_conv5_weights_ref_'+str(train_ref_ori)+'_sf_'+str(train_sf)+'_sep_'+str(train_sep)+'_lr_0.0001_trial_'+str(trial_num), weights_conv5, allow_pickle=True)
            
            # Free memory before transfer evaluation
            del weights_fc
            if model_name == 'AlexNet' or model_name == 'AlexNet_skip':
                del weights_conv1, weights_conv2, weights_conv3, weights_conv4, weights_conv5
                del final_conv1_weights, final_conv2_weights, final_conv3_weights, final_conv4_weights, final_conv5_weights
            del final_fc_weights, init_fc_weights
            del train_dataloader, test_dataloader
            del train_grating_dataset, test_grating_dataset
            del train_accuracies, test_accuracies, summed_train_accuracies, summed_test_accuracies
            del train_confidence, test_confidence, train_noisy_confidence, test_noisy_confidence
            torch.cuda.empty_cache()
            
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
                    test_acc = [output_labels[i] == test_labels[i].item() for i in range(0,len(output_labels))]
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
                
                # Clean up transfer data
                del transfer_dataset, transfer_dataloader
                del transfer_losses, transfer_accuracies, transfer_confidence, transfer_noisy_confidence
                torch.cuda.empty_cache()
                
                if not sequential:
                    count += 1
                test_count += 1

                dir_count += 1

