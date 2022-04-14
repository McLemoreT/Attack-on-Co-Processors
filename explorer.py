#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 13:24:29 2022

@author: tyler
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import _thread

from PIL import Image

import torchvision
import TorchUtils as TorchUtils
from testing import isGoodPlace


import math
class quarry:
    
    def editImage(location, diff, image): #location is location list
        #diff is distance to perturb 
        #image is the image
        for cord in location:
            #Go to the location specified by cord in image, and add diff
            
            image[0][cord[1]][cord[0]] = image[0][cord[1]][cord[0]] + diff 
            
        return image

    
    def binaryString(number):
        return format(number, 'b') #Convert an into to a string of 0's and 1's
    
    def rankFix(image):
        while(image.dim() > 3):
            image = image[0]
        return image
    
    def makeCoordinates(edit_list,image):
        edit_list = edit_list [::-1] #Invert the string edit_list
        itter = 0
        
        arr = np.zeros((edit_list.count('1'),2)) #Create an array is as big as the number of 1's in edit_list
        
        width = image.size(dim=2)# + 1#Subtract 1 because we are counting 0 as the first number
        i = 0
        for index in edit_list:
            if(index == '1'):
                arr[i][0] = itter % width
                arr[i][1] = math.floor(itter / width)
                i = i + 1
            itter = itter + 1
        return arr.astype(int)
            

    def displayImage (image): #Just a quick function to display an image
         plt.figure()
         plt.imshow(image.reshape((image.size(dim=2), image.size(dim=2)))) #shows it
         plt.show()
         plt.close()
         
    def saveImage (image, modifier):
        name = "images/" + str(modifier) + ".png"
        plt.imsave(name, image.reshape((image.size(dim=2), image.size(dim=2))))
    
    def QuarrySave(image, #The image we are testing
                   iterations, #How many perturbed images are we making?
                   starting_number, #What number are we starting at for quarry?
                   model, #What is the model
                   patchedModel, #What is the memtorch model
                   actual_class, #What is the actual classification of image
                   modifier, #Naming modifier for saving the image, typically to identify the perturbed images as a subset of the original
                   ending_number = -1): #Max number defaults to -1 so we can calculate the actual max number
        #Iterations are the number of times this should run
        #Probably shouldset an upper limit when making quarry images
        dims = image.dim();
        if ending_number == -1: #If the max number isn't set, set it to the true max number
            ending_number = pow(2,image.size(dim=dims) * image.size(dim=dims - 1))
        
        for x in range(starting_number, ending_number, math.floor(ending_number/iterations)):
                       #Generate information for new image
                       bin_string = quarry.binaryString(x) 
                       cords = quarry.makeCoordinates(bin_string, image)
                       #Clone and detach old image
                       new_image = image.clone().detach()
                       #Generate new image
                       new_image = quarry.editImage(cords, TorchUtils.getNormParam(image)[2], new_image)
                       
                       if(isGoodPlace(model, patchedModel, new_image, actual_class)):
                           #save
                           name = "images/" + modifier + "---" + str(x) + ".png"
                           plt.imsave(name, new_image.reshape((new_image.size(dim=2), new_image.size(dim=2))))
                       else:
                           #Don't save
                           1+1
                       
        plt.close()

         
        
        
        
if __name__ == '__main__': #Basically everything here is just test functions
    test_tensor = torch.FloatTensor([[[[ 0.0000e+00,  0.0000e+00,  3.6297e-04,  5.9150e-05, -7.3652e-03,
                -5.7948e-03,  1.8529e-02,  2.1448e-03,  3.8571e-02,  4.1277e-03,
                2.0284e-02,  1.6984e-02,  2.8106e-02,  1.6503e-02,  1.7238e-03,
                8.1881e-04, -2.9310e-03,  3.7401e-03,  0.0000e+00,  0.0000e+00,
                0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
                0.0000e+00,  0.0000e+00,  0.0000e+00],
              [ 0.0000e+00,  0.0000e+00,  1.7453e-04,  6.4883e-04, -1.7059e-03,
                -1.9587e-03,  3.4526e-02,  2.2776e-02,  3.7630e-02,  1.5577e-02,
                6.7939e-02,  3.3312e-02,  4.5620e-02,  2.3764e-02,  2.1300e-02,
                1.5785e-02,  2.0970e-02,  6.5018e-03,  0.0000e+00,  0.0000e+00,
                0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
                0.0000e+00,  0.0000e+00,  0.0000e+00],
              [ 5.5677e-03, -3.1520e-03,  1.3013e-02, -2.7989e-03, -6.0178e-03,
                1.6033e-02,  3.6427e-02,  2.8266e-03,  5.1154e-02,  3.2437e-02,
                9.9915e-02,  6.2299e-02,  8.9790e-02,  5.0260e-02,  7.5151e-02,
                2.6451e-02,  4.2801e-02,  7.6311e-03,  7.0732e-03,  9.9554e-03,
                -1.6461e-03, -9.7940e-03, -4.1359e-03,  4.3516e-03,  1.0698e-03,
                -3.5694e-03,  8.1392e-04,  1.1841e-03],
              [ 5.1917e-04,  1.5127e-03,  2.4798e-02,  1.0095e-02, -2.1534e-02,
                9.7298e-03,  3.5112e-02,  8.4128e-03,  4.8651e-02,  5.5805e-02,
                1.1889e-01,  1.0826e-01,  1.1800e-01,  3.8198e-02,  4.3490e-02,
                -1.4322e-02,  6.3635e-03, -1.5937e-02, -4.3385e-03, -5.4400e-03,
                -3.2664e-03, -1.7413e-02,  1.3769e-02,  1.3485e-02,  1.7403e-02,
                4.8174e-03,  4.0627e-03,  9.4005e-04],
              [-1.6646e-03, -2.5733e-03,  1.8800e-02,  1.3897e-02, -1.8027e-02,
                1.5638e-02,  2.9052e-03,  1.1643e-02, -3.6704e-02,  5.3657e-02,
                6.9876e-02,  6.4500e-02,  6.8915e-02,  7.7904e-02, -2.2439e-02,
                -2.4845e-02, -3.6822e-02, -2.4329e-02, -2.4216e-02,  2.6544e-03,
                2.2752e-03, -1.7750e-02,  2.7619e-02,  3.8747e-02,  2.4435e-02,
                -6.5561e-03,  7.4059e-03,  6.3429e-03],
              [ 1.5344e-03, -2.7699e-03,  3.0967e-02,  1.2306e-02, -2.1385e-02,
                3.1186e-02,  2.4281e-02, -9.7743e-03,  4.5341e-03,  7.4468e-03,
                4.6428e-02,  7.4939e-02,  6.8185e-02,  1.4321e-01,  9.1358e-02,
                -1.1648e-02, -6.1685e-02,  9.5873e-03, -3.3170e-02,  5.2439e-02,
                1.1413e-01,  1.7083e-01,  8.2548e-01,  8.7374e-01,  5.5492e-01,
                4.2201e-02,  5.0412e-02,  9.8370e-03],
              [-9.3875e-03, -5.0505e-03,  1.2372e-02,  2.4252e-02, -6.4166e-02,
                -1.6432e-02,  4.8244e-02, -2.4050e-02,  5.3988e-02, -5.7749e-03,
                -2.7466e-02, -3.5688e-02,  1.1284e-01,  1.0985e-01,  7.7722e-02,
                1.9318e-02, -9.1524e-03, -3.2199e-02, -1.1477e-02,  7.3427e-01,
                1.0266e+00,  9.6132e-01,  9.4987e-01,  9.8926e-01,  9.5409e-01,
                -1.6352e-02,  5.9911e-02, -2.9789e-02],
              [ 5.1421e-03, -4.5196e-03,  9.0625e-03,  2.5067e-02, -7.9715e-02,
                3.6303e-02,  5.2615e-03, -8.9568e-02, -8.4342e-02, -3.1351e-02,
                -5.1474e-03,  1.7051e-02,  9.7830e-02,  6.6906e-01,  8.1733e-01,
                8.2104e-01,  8.4817e-01,  8.5044e-01,  8.3551e-01,  9.9620e-01,
                9.8935e-01,  9.4399e-01,  1.0026e+00,  1.0126e+00,  6.0942e-01,
                -1.1256e-01,  1.2948e-02, -4.1531e-04],
              [-7.8266e-03, -7.3962e-03, -3.7352e-02,  9.3455e-03, -1.1243e-01,
                -7.1530e-02, -3.4422e-02, -3.5559e-02, -1.3469e-01, -9.4971e-03,
                -3.9350e-02, -1.5194e-01,  7.3918e-01,  8.9694e-01,  9.7575e-01,
                1.0013e+00,  9.9327e-01,  9.4831e-01,  6.6189e-01,  4.4405e-01,
                4.1085e-01,  5.0653e-01,  3.9627e-01,  1.1284e-01, -4.7402e-02,
                -1.3413e-01, -8.3090e-02,  3.7808e-03],
              [-2.7712e-03,  8.1698e-04,  5.8063e-03,  1.8835e-02,  2.0199e-02,
                -1.9872e-02, -1.1466e-02,  7.6780e-03, -3.7744e-02,  1.4511e-02,
                -2.7530e-02,  2.1775e-01,  9.9079e-01,  9.9694e-01,  1.0418e+00,
                8.8904e-01,  2.1810e-01,  1.0160e-01,  7.3356e-03,  8.1695e-02,
                5.9508e-02,  1.2438e-01,  8.5856e-04,  1.6240e-02, -6.7666e-02,
                -3.0882e-02, -6.0731e-02,  2.0510e-02],
              [-7.0312e-03,  1.9863e-03,  6.5188e-03,  2.1921e-02,  2.7378e-02,
                5.2359e-03,  2.9029e-02,  4.1995e-02, -7.7384e-02, -6.2981e-03,
                3.2725e-01,  9.2410e-01,  1.0153e+00,  9.8991e-01,  1.0346e+00,
                1.8728e-01,  5.7878e-02,  7.0645e-02,  8.4995e-02,  2.0340e-01,
                1.1053e-01,  3.5238e-02, -9.7105e-02, -6.9018e-02, -5.0394e-02,
                -1.1024e-02,  1.3121e-02, -5.3763e-03],
              [-2.9409e-02,  4.7546e-03,  2.5202e-02,  2.0861e-02,  1.8768e-02,
                4.0758e-02,  3.7398e-02, -5.7556e-02, -1.0308e-01, -2.4425e-02,
                8.1225e-01,  9.9732e-01,  9.5890e-01,  9.6469e-01,  2.2403e-01,
                4.5603e-02,  3.4370e-02,  3.3318e-04,  5.5606e-02,  1.0364e-01,
                1.9864e-02, -3.9973e-03, -5.4743e-02, -3.5536e-02, -6.8377e-02,
                -5.5834e-02,  2.1467e-03,  2.8855e-02],
              [-2.0416e-02, -3.4826e-03,  8.7505e-03,  2.3258e-02,  7.5449e-03,
                4.2020e-02,  1.6043e-02, -6.3704e-02, -1.4945e-02,  3.3485e-01,
                9.3249e-01,  9.5255e-01,  9.4542e-01,  8.7730e-01,  2.0498e-01,
                1.1746e-01,  1.0837e-01,  1.1173e-01,  9.9054e-02,  1.0949e-01,
                1.3575e-02, -7.2697e-04, -2.3119e-02, -2.9994e-02, -9.1032e-02,
                -7.7160e-02, -1.9273e-02,  1.6582e-02],
              [-1.9556e-03,  5.6624e-03,  1.2854e-02,  1.0222e-02, -2.2779e-03,
                -2.1075e-03, -1.5199e-02, -1.2279e-02,  5.9096e-02,  1.5332e-01,
                8.8297e-01,  9.6567e-01,  9.6501e-01,  9.9148e-01,  9.7175e-01,
                7.5226e-01,  2.2963e-01,  1.1801e-01,  1.1286e-01,  1.2542e-01,
                1.0120e-01,  9.9235e-02,  7.5776e-02,  3.9291e-04, -8.2820e-03,
                2.9166e-02,  1.9369e-02,  1.6992e-02],
              [ 6.0872e-04,  3.3734e-03,  1.9949e-02,  1.7852e-02, -9.9467e-03,
                -7.7070e-03, -3.5531e-02,  2.4939e-02, -9.6762e-03,  3.5622e-02,
                2.2581e-01,  7.6592e-01,  9.6657e-01,  9.9309e-01,  9.9467e-01,
                9.5636e-01,  9.1028e-01,  7.0311e-01,  6.2605e-02,  1.2394e-01,
                9.2689e-02,  6.8957e-02,  7.4639e-02,  2.6809e-02, -2.8313e-02,
                -1.1514e-03,  3.1431e-02,  1.7656e-02],
              [ 5.8261e-04,  2.9209e-03, -5.8631e-04,  9.4149e-03, -3.8030e-02,
                -7.4133e-03, -4.2580e-02, -3.4970e-02, -7.5131e-02, -2.5814e-02,
                -3.4901e-02,  1.5022e-02,  4.4329e-02,  1.0027e-02,  3.8412e-01,
                9.4036e-01,  9.2581e-01,  9.7362e-01,  3.5286e-01,  5.3610e-02,
                5.6850e-02,  4.7466e-02,  3.9364e-02,  1.8641e-02, -1.4100e-02,
                -5.0343e-03, -1.0444e-02,  3.7185e-03],
              [ 4.0546e-03,  1.8472e-03,  2.6033e-03,  6.0833e-03, -1.8990e-02,
                -4.4197e-03, -1.0245e-02, -4.0152e-03, -5.1062e-02,  1.3917e-02,
                -1.1428e-02,  3.8854e-02,  4.5624e-02,  3.9148e-02,  4.2019e-02,
                2.5638e-01,  9.1777e-01,  9.4981e-01,  4.6523e-01,  2.7119e-02,
                1.2882e-02,  2.0658e-02,  3.9588e-02,  3.0269e-02, -3.5911e-03,
                -2.0368e-02, -1.1338e-02,  2.0828e-03],
              [ 5.0551e-04,  3.6518e-03, -1.1982e-03,  4.4586e-03, -6.6168e-03,
                -8.5884e-03, -1.4546e-02, -2.8531e-02, -4.4231e-02, -2.7284e-02,
                -4.7783e-02, -1.6225e-02, -3.5725e-02, -1.4465e-03,  3.1396e-02,
                4.7422e-02,  8.1880e-01,  9.7229e-01,  2.8815e-01, -7.2587e-04,
                2.7105e-02,  4.2577e-02,  9.5639e-03,  3.4834e-02, -7.4790e-03,
                -7.7650e-03, -1.8197e-02, -1.5291e-03],
              [ 0.0000e+00,  0.0000e+00, -1.3069e-04, -3.3443e-04, -1.5498e-02,
                -5.8366e-03, -1.7709e-02, -5.5314e-03, -7.3526e-03,  1.1547e-03,
                -2.9584e-02, -5.9446e-02, -4.9028e-02, -7.1042e-02, -1.9577e-02,
                5.7334e-01,  1.0064e+00,  9.3541e-01,  6.1626e-02,  4.7216e-02,
                3.1546e-02, -2.4891e-03, -1.9494e-02, -3.2950e-03, -9.6357e-03,
                -1.6684e-02, -3.1557e-02, -3.8903e-03],
              [ 0.0000e+00,  0.0000e+00,  6.8852e-04,  1.2602e-03, -5.6035e-03,
                4.1735e-03, -2.4995e-02, -1.6694e-02, -1.7446e-02, -1.6147e-02,
                -1.6563e-02,  2.2789e-03, -3.4197e-02,  3.0163e-02,  4.0259e-01,
                9.3214e-01,  1.0033e+00,  5.0718e-01,  3.6665e-02,  3.1612e-02,
                2.8773e-02,  2.3458e-02, -1.7237e-02,  1.5760e-02, -3.4802e-03,
                -7.8866e-03, -1.7400e-02,  1.5904e-03],
              [ 0.0000e+00,  0.0000e+00,  1.4101e-03,  1.5306e-03,  1.3546e-03,
                9.6294e-04, -4.5665e-03,  4.8510e-03,  2.9116e-03,  5.0550e-03,
                -4.6298e-03, -1.7259e-03,  4.9241e-02,  5.5820e-01,  9.8840e-01,
                9.9726e-01,  9.0140e-01,  1.3147e-01,  1.4503e-02,  9.7282e-03,
                -2.3729e-02, -1.7000e-02, -4.8762e-02, -2.0059e-02, -9.2015e-03,
                -3.8783e-03, -1.3310e-02, -1.2950e-03],
              [ 0.0000e+00,  0.0000e+00,  2.4306e-04,  6.8951e-04,  1.5282e-03,
                4.8188e-01,  1.4567e-01, -1.6221e-02,  5.6332e-02, -5.9937e-02,
                -7.3735e-02,  2.1550e-01,  9.2931e-01,  9.8076e-01,  9.6413e-01,
                6.5643e-01,  7.3736e-02,  4.3568e-02, -3.2039e-02,  8.7170e-03,
                -5.1834e-03,  2.6354e-03, -2.7418e-02, -5.9815e-03,  2.1732e-03,
                -2.3385e-04, -2.3146e-03, -1.3351e-03],
              [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
                8.0784e-01,  6.0826e-01,  1.7676e-01,  1.6242e-01,  1.8463e-01,
                7.9355e-01,  1.0728e+00,  9.8713e-01,  8.1976e-01,  2.8068e-01,
                6.0263e-02,  4.0469e-02,  5.4502e-02, -1.2111e-02, -2.0788e-03,
                -3.1225e-03, -2.3776e-03, -2.8786e-02, -1.6462e-03, -6.3596e-03,
                4.6426e-03, -4.3739e-03,  0.0000e+00],
              [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
                6.7451e-01,  9.9282e-01,  1.0070e+00,  9.8481e-01,  1.0244e+00,
                1.0709e+00,  9.0006e-01,  6.1690e-01,  7.0018e-02,  1.7043e-02,
                5.0477e-02,  4.1407e-02,  1.8124e-02,  4.7666e-03,  1.3104e-02,
                7.4653e-04, -8.5605e-04, -9.3616e-03, -1.1280e-03, -1.6951e-03,
                -2.3821e-03, -3.1762e-03,  0.0000e+00],
              [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
                3.2157e-01,  6.7467e-01,  1.0690e+00,  1.0212e+00,  7.6344e-01,
                4.1132e-01,  5.3002e-02,  2.0168e-02,  3.7563e-02,  2.9311e-02,
                3.3016e-02, -6.1551e-03, -6.7824e-04, -1.4365e-02, -2.0919e-03,
                -1.7093e-03,  1.2927e-02, -1.8252e-03,  1.1703e-03, -2.8046e-03,
                -2.7674e-04, -2.5066e-03,  0.0000e+00],
              [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
                0.0000e+00,  7.0321e-04, -4.5061e-02, -5.6091e-03, -2.8981e-02,
                -4.7250e-02, -7.3050e-02,  1.7234e-02,  3.7980e-02,  2.8405e-02,
                2.9120e-02,  1.8162e-02,  2.2315e-02, -2.8052e-03,  1.3155e-02,
                6.4031e-03,  1.5260e-03, -2.0332e-03, -3.4034e-04, -5.2117e-04,
                -6.1359e-04,  0.0000e+00,  0.0000e+00],
              [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
                0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
                2.6319e-02,  2.0106e-02,  1.9277e-02,  2.7065e-02,  3.2077e-03,
                2.0805e-02,  9.0117e-03,  1.8477e-02, -4.6866e-03,  9.5600e-03,
                -3.1677e-03,  1.3694e-03, -5.0295e-03,  0.0000e+00,  0.0000e+00,
                0.0000e+00,  0.0000e+00,  0.0000e+00],
              [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
                0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
                1.1657e-02, -1.4170e-03,  3.4763e-03,  8.6602e-03,  1.5003e-02,
                9.8687e-03, -2.5351e-03,  4.0140e-03,  1.8074e-03,  8.9458e-04,
                -2.3539e-04,  2.9761e-05, -1.7513e-04,  0.0000e+00,  0.0000e+00,
                0.0000e+00,  0.0000e+00,  0.0000e+00]]]])
    # test.displayImage(test_tensor)
    # print(test.binaryConverter(7))
    # print(test.getCoordinates(test.binaryConverter(7), test_tensor))
    
    # small = torch.FloatTensor([[[[0,0,0,0],[1,1,1,1],[0,0,0,0],[0,1,0,1]]]])
    # print(small)
    
    # binstring = test.binaryString(17)
    # cords = test.makeCoordinates(binstring, small)
    # oldsmall = small.clone().detach()
    
    # test.editImage(cords, 0.5, small)
    # test.displayImage(small)
    # test.displayImage(oldsmall)
    
    # binstring = test.binaryString(624109300249999999999999999999999944600573)
    # cords = test.makeCoordinates(binstring, test_tensor)
    # test.displayImage(test_tensor)
    # test.editImage(cords, 0.5, test_tensor)
    # test.displayImage(test_tensor)
    
    start = time.time()
    number = 1
    current_num = 0
    while number < 100000:
        current_num = current_num + number
        print(current_num)
        binstring = quarry.binaryString( current_num)
        cords = quarry.makeCoordinates(binstring, test_tensor)
        newtest = test_tensor.clone().detach()
        quarry.editImage(cords, 0.5, newtest)
        quarry.displayImage(newtest)
        
       # newtest = test_tensor.clone().detach()
        #quarry.getPerturbedImage(newtest, number)
        
        #name = "images/" + str(number) + ".png"
        #plt.imsave(name, newtest.reshape((newtest.size(dim=2), newtest.size(dim=2))))
        

        
        #test.displayImage(newtest)
        
        #_thread.start_new_thread(test.thread_test, (number, test_tensor))
        #test.thread_test(number, test_tensor)
        
        #Threading test 1
        #_thread.start_new_thread(test.editImage , (test.makeCoordinates(binstring, test_tensor), 0.5, test_tensor.clone().detach()))
        
        #Display image threaded
        #test.displayImage(test.editImage(test.makeCoordinates(binstring, test_tensor), 0.5, test_tensor.clone().detach()))
        
        number = number + 1

        
        
    plt.close()
    end = time.time()
    timer = end - start
    print(timer)
    
    
    # test.displayImage(small)
    # edits = test.binaryConverter(16)
    # cords = test.getCoordinates(edits, small)
    # new_small = test.editImage(cords, 0.5, small)
    # test.displayImage(new_small)
