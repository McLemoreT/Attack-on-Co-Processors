#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 12:45:09 2022

@author: tyler
"""
import torch
import TorchUtils
import memtorch
import deepfool

#Other Files
from TorchUtils import goodPerturb
import explorer as explorer
    

def QuarryTest(polyset, model, patchedModel, fool_set):
    counter = 0
    isGoodPlace_Loader = torch.utils.data.DataLoader(
        fool_set, batch_size=100, shuffle=True, num_workers=8
    )
    while counter < 5: #Number of batches to go through
        images, label = next(iter(isGoodPlace_Loader)) #A loader iterator returns a tensor of images, and their
                                                #labels
        for i in range(0, len(label)):
            actual_class, label_software, label_memristor, count, hash_val, pert_image = goodPerturb(model, patchedModel, images[i], label[i])
           
            if pert_image is not None:
                pert_image = explorer.quarry.rankFix(pert_image)
                explorer.quarry.QuarrySave(pert_image, 100, 0, model, patchedModel, label_software, str(counter) + " " + str(i), ending_number = 10000)

def goodPerturbTest(fool_set, model, patchedModel):
    counter = 0
    good_data = []
    new_loader = torch.utils.data.DataLoader(
        fool_set, batch_size=100, shuffle=True, num_workers=8
    )
    while counter < 5: #Number of batches to go through
        images, label = next(iter(new_loader)) #A loader iterator returns a tensor of images, and their
                                                #labels
        for i in range(0, len(label)):
            
            good_data.append(goodPerturb(model, patchedModel, images[i], label[i]))