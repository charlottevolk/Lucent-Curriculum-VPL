# train the alexnet model (from test_alexnet.py) on the grating dataset
# load the model and the dataset

if __name__ == '__main__':

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torchvision
    from torchvision import transforms, utils
    from torchvision import datasets, models, transforms
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
    from dataset_activations import GratingDataset
    from transforms import GaussianNoise
    from activation_alexnet import AlexNet
    from statistics import mean

    # set the device to use
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # set the parameters for the dataset
    root_dir = 'StimulusImages/imagenet_100_1000/' # Put the shared 1000 images from the COCO dataset here
    ref_orientation = 0
    separation_angle = 10
    contrast = 1
    phase = 0
    spatial_freq = 10
    num_images = 100

    # set the parameters for the model
    num_classes = 2
    batch_size = 1
    num_epochs = 1
    learning_rate = 0.0001

    stimulus_noise_sd = 0.0#25/255 # Caffe does noise in pixel values, torch does it in 0-1 tensor space
    # confidence_noise_sd = 0.3#25/255

    # set the transforms
    data_transforms = transforms.Compose([
        transforms.Resize((227,227)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        GaussianNoise(0, stimulus_noise_sd), # STANDARD DEVIATION OF GAUSSIAN NOISE
    ])
    
    train_dir = [root_dir,0,0.05,0.5]
    
    base_dir = 'saved_outputs/'
    save_dir = base_dir+'skip_activations_two_step_models_imagenet_100/'
    
    trial_nums = range(1,21)

    for trial_num in trial_nums:

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

        #For new models
        model.append(AlexNet())
        alexnet = torchvision.models.alexnet(pretrained=True)
        copy_weights(model[count], alexnet)
        model[count] = model[count].to(device)


        model[count].conv1.register_forward_hook(getActivation("c1"))
        model[count].pool1.register_forward_hook(getActivation("p1"))
        model[count].conv2.register_forward_hook(getActivation("c2"))
        model[count].pool2.register_forward_hook(getActivation("p2"))
        model[count].conv3.register_forward_hook(getActivation("c3"))
        model[count].conv4.register_forward_hook(getActivation("c4"))
        model[count].conv5.register_forward_hook(getActivation("c5"))
        model[count].pool3.register_forward_hook(getActivation("p3"))

        # Get parameters
        train_root_dir = train_dir[0]
        train_ref_ori = train_dir[1]
        train_sf = train_dir[2]
        train_sep = train_dir[3]

        #plt.imshow(train_ref_img.permute(1,2,0))

        # load the dataset
        train_grating_dataset = GratingDataset(train_root_dir, transform=data_transforms, ref_orientation=ref_orientation, separation_angle=separation_angle, contrast=contrast, phase=phase, spatial_freq=spatial_freq, num_images=num_images)

        # create the dataloader
        train_dataloader = DataLoader(train_grating_dataset, batch_size=1, shuffle=True, num_workers=4)

        iter_count = 0

        # train the model

        activations_all = []
            
        for epoch in range(num_epochs):
            for (train_i, train_data) in enumerate(train_dataloader):
                activations = []

                # set model to eval mode
                model[count].eval()

                train_inputs, train_labels = train_data
                train_inputs = train_inputs.to(device)
                train_labels = train_labels.to(device)

                train_outputs = model[count](train_inputs, train_sep)

                activations.append(np.array(torch.tensor(activation["c1"].cpu().numpy()).flatten()))
                activations.append(np.array(torch.tensor(activation["p1"].cpu().numpy()).flatten()))
                activations.append(np.array(torch.tensor(activation["c2"].cpu().numpy()).flatten()))
                activations.append(np.array(torch.tensor(activation["p2"].cpu().numpy()).flatten()))
                activations.append(np.array(torch.tensor(activation["c3"].cpu().numpy()).flatten()))
                activations.append(np.array(torch.tensor(activation["c4"].cpu().numpy()).flatten()))
                activations.append(np.array(torch.tensor(activation["c5"].cpu().numpy()).flatten()))
                activations.append(np.array(torch.tensor(activation["p3"].cpu().numpy()).flatten()))
                
                iter_count += 1
                activations_all.append(np.concatenate(activations))
            
        
        print('Finished Training & Testing')

        np.save(save_dir+'activations/100_imgs_all_activations_no_noise_1000_imagenet_'+str(train_ref_ori)+'_sf_'+str(train_sf)+'_sep_'+str(train_sep)+'_lr_0.0001_model_trial_'+str(trial_num)+'_batch_size_1.npy', activations_all, allow_pickle=True)
